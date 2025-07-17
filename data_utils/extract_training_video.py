import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm
import logging  # 导入 logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_utils.v2a_utils.vggsound_224 import VGGSound
from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils
import torchaudio
from einops import rearrange
from torch.utils.data.dataloader import default_collate


# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

def main(args):
    rank = int(os.environ["RANK"])  # 从环境变量中获取 rank
    world_size = int(os.environ["WORLD_SIZE"])  # 从环境变量中获取 world size
    setup(rank, world_size)

    dataset = VGGSound(
        root=args.root,
        tsv_path=args.tsv_path,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        audio_samples=args.audio_samples,
        start_row=args.start_row,
        end_row=args.end_row,
        save_dir=args.save_dir
    )
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    # 使用 DataLoader 加载数据集，增加 batch_size 和 num_workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, sampler=train_sampler, num_workers=8, drop_last=False,collate_fn=error_avoidance_collate)

    feature_extractor = FeaturesUtils(
        vae_ckpt=args.vae_ckpt,
        vae_config=args.vae_config,
        enable_conditions=True,
        synchformer_ckpt=args.synchformer_ckpt
    ).eval().cuda(rank)

    # 使用 DistributedDataParallel 支持多显卡
    feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[rank])

    # 使用 tqdm 显示进度条
    for i, data in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):
        # 使用 torch.no_grad() 来加快推理速度
        ids = data['id']  # 获取当前批次的所有 ID
        with torch.no_grad():
            audio = data['audio'].cuda(rank, non_blocking=True)

            output = {
                'caption': data['caption'],
                'caption_cot': data['caption_cot']
            }
            # logging.info(f'Processing batch {i} with IDs: {ids}')  # 添加日志记录

            latent = feature_extractor.module.encode_audio(audio)
            output['latent'] = latent.detach().cpu()

            clip_video = data['clip_video'].cuda(rank, non_blocking=True)
            # logging.info(f'Processing batch {i} with shape: {clip_video.shape}')  # 添加日志记录
            clip_features = feature_extractor.module.encode_video_with_clip(clip_video)
            output['metaclip_features'] = clip_features.detach().cpu()

            sync_video = data['sync_video'].cuda(rank, non_blocking=True)
            sync_features = feature_extractor.module.encode_video_with_sync(sync_video)
            output['sync_features'] = sync_features.detach().cpu()

            caption = data['caption']
            metaclip_global_text_features, metaclip_text_features = feature_extractor.module.encode_text(caption)
            output['metaclip_global_text_features'] = metaclip_global_text_features.detach().cpu()
            output['metaclip_text_features'] = metaclip_text_features.detach().cpu()

            caption_cot = data['caption_cot']
            t5_features = feature_extractor.module.encode_t5_text(caption_cot)
            output['t5_features'] = t5_features.detach().cpu()


            # 保存每个样本的输出
            for j in range(audio.size(0)):  # 遍历当前批次的每个样本
                sample_output = {
                    'id': ids[j],
                    'caption': output['caption'][j],
                    'caption_cot': output['caption_cot'][j],
                    'latent': output['latent'][j],
                    'metaclip_features': output['metaclip_features'][j],
                    'sync_features': output['sync_features'][j],
                    'metaclip_global_text_features': output['metaclip_global_text_features'][j],
                    'metaclip_text_features': output['metaclip_text_features'][j],
                    't5_features': output['t5_features'][j],
                }
                torch.save(sample_output, f'{save_dir}/{ids[j]}.pth')

        ## test the sync between videos and audios
        # torchaudio.save(f'input_{i}.wav',data['audio'],sample_rate=44100)
        # recon_audio = feature_extractor.decode_audio(latent)
        # recon_audio = rearrange(recon_audio, "b d n -> d (b n)")
        # id = data['id']
        # torchaudio.save(f'recon_{i}.wav',recon_audio.cpu(),sample_rate=44100)
        # os.system(f'ffmpeg -y -i dataset/vggsound/video/train/{id}.mp4 -i recon_{i}.wav -t 9 -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental -shortest out_{i}.mp4')
        # os.system(f'ffmpeg -y -i dataset/vggsound/video/train/{id}.mp4 -i input_{i}.wav -t 9 -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental -shortest input_{i}.mp4')
    cleanup()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Video Training Latents')
    parser.add_argument('--root', type=str, default='dataset/vggsound/video/test', help='Root directory of the video dataset')
    parser.add_argument('--tsv_path', type=str, default='dataset/vggsound/split_txt/test_caption.csv', help='Path to the TSV file')
    parser.add_argument('--save-dir', type=str, default='dataset/vggsound/video_text_latents/test', help='Save Directory')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the audio')
    parser.add_argument('--duration_sec', type=float, default=9.0, help='Duration of the audio in seconds')
    parser.add_argument('--audio_samples', type=int, default=397312, help='Number of audio samples')
    parser.add_argument('--vae_ckpt', type=str, default='ckpts/vae.ckpt', help='Path to the VAE checkpoint')
    parser.add_argument('--vae_config', type=str, default='ThinkSound/configs/model_configs/stable_audio_2_0_vae.json', help='Path to the VAE configuration file')
    parser.add_argument('--synchformer_ckpt', type=str, default='ckpts/synchformer_state_dict.pth', help='Path to the Synchformer checkpoint')
    parser.add_argument('--start-row', type=int, default=0, help='start row')
    parser.add_argument('--end-row', type=int, default=None, help='end row')

    args = parser.parse_args()

    # 直接使用 torch.distributed.launch 启动
    main(args=args)  # 这里的 rank 需要在命令行中指定

