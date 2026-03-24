import argparse
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import sys
import torch
import logging
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
import time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils
from data_utils.v2a_utils.thinksound_288_al import VGGSound
from torch.utils.data.dataloader import default_collate

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None  # 或 return {}
    return default_collate(batch)

def process_batch(data, model, rank, inference_mode=False):
    output = {
        'caption_cot': data['caption_cot'],
        'latent': [],
        'global_video_features': [],
        'video_features': [],
        'global_text_features': [],
        'text_features': [],
        'sync_features': [],
    }
    #start_time = time.time()
    
    with torch.no_grad():
        
        text_features = model.module.encode_t5_text(data['caption_cot'])
        output['text_features'] = text_features.detach().cpu().numpy()
        if not inference_mode:
            latent = model.module.encode_audio(data['audio'].cuda(rank, non_blocking=True))
            output['latent'] = latent.detach().cpu().numpy()
        else:
            output['latent'] = [None] * len(text_features)

        video_feat,frame_embed,_,text_feat= model.module.encode_video_and_text_with_videoprism(data['clip_video'], data['caption_cot'])

        output['global_video_features'].append(np.array(video_feat))
        output['video_features'].append(np.array(frame_embed))
        output['global_text_features'].append(np.array(text_feat))

        sync_video = data['sync_video'].cuda(rank, non_blocking=True)
        sync_features = model.module.encode_video_with_sync(sync_video)
        output['sync_features'] = sync_features.detach().cpu().numpy()



    return output


def save_outputs(output, ids, save_dir, add_audio_path=None, add_video_path=None):
    for i, sample_id in enumerate(ids):
        np.savez(
            os.path.join(save_dir, f"{sample_id}.npz"),
            id=sample_id,
            audio_path=os.path.join(add_audio_path,f"{sample_id}.wav") if add_audio_path is not None else None,
            video_path=os.path.join(add_video_path,f"{sample_id}.mp4") if add_video_path is not None else None,
            caption_cot=output['caption_cot'][i],
            latent=output['latent'][i],
            global_video_features=output['global_video_features'][i],
            video_features=output['video_features'][i],
            global_text_features=output['global_text_features'][i],
            text_features=output['text_features'][i],
            sync_features=output['sync_features'][i],
        )


def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_distributed(rank, world_size)

    dataset = VGGSound(
        root=args.root,
        tsv_path=args.tsv_path,
        sample_rate=args.sample_rate,
        start_row=args.start_row,
        end_row=args.end_row,
        save_dir=args.save_dir,
        inference_mode = args.inference_mode
    )

    os.makedirs(args.save_dir, exist_ok=True)

    sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=32,
                            drop_last=False, collate_fn=error_avoidance_collate, pin_memory=True)

    model = FeaturesUtils(
        vae_ckpt=args.vae_ckpt if not args.inference_mode else None,
        vae_config=args.vae_config,
        enable_conditions=True,
        synchformer_ckpt=args.synchformer_ckpt
    )
    model = model.eval().cuda(rank)

    torch.compile(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    for data in tqdm(dataloader, desc="Processing", unit="batch"):
        if data is None:
            continue
        ids = data['id']
        try:
            output = process_batch(data, model, rank, args.inference_mode)
            save_outputs(output, ids, args.save_dir, args.add_audio_path, args.add_video_path)
        except Exception as e:
            logging.error(f"Error processing sample IDs {ids}: {e}")

    cleanup_distributed()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Extract Video Training Latents')
    parser.add_argument('--root', default='videos')
    parser.add_argument('--tsv_path', default='cot_coarse/cot.csv')
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Audio sample rate')
    parser.add_argument('--vae_ckpt', type=str, default='ckpts/vae.ckpt', help='Path to the VAE checkpoint')
    parser.add_argument('--vae_config', type=str, default='PrismAudio/configs/model_configs/stable_audio_2_0_vae.json', help='Path to the VAE configuration file')
    parser.add_argument('--synchformer_ckpt', type=str, default='ckpts/synchformer_state_dict.pth', help='Path to the Synchformer checkpoint')
    parser.add_argument('--start-row', '-s', type=int, default=0, help='Start row index')
    parser.add_argument('--end-row', '-e', type=int, default=None, help='End row index')
    parser.add_argument('--add_audio_path', default=None, 
                    help='Provide the original audio file path required for ITD reward in GRPO')
    parser.add_argument('--add_video_path', default=None, 
                        help='Provide the video path file required for Synchformer reward in GRPO')
    parser.add_argument('--inference_mode', default=False, help='inference_mode')

    args = parser.parse_args()
    main(args)
