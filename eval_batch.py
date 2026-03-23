from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import re
import torch
import torchaudio
from lightning.pytorch import seed_everything
import random
from datetime import datetime
import numpy as np

from PrismAudio.data.datamodule import DataModule
from PrismAudio.models import create_model_from_config
from PrismAudio.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from PrismAudio.inference.sampling import sample, sample_discrete_euler
from pathlib import Path
from tqdm import tqdm


def predict_step(diffusion, batch, diffusion_objective, device='cuda:0'):
    diffusion = diffusion.to(device)

    reals, metadata = batch
    ids = [item['id'] for item in metadata]
    batch_size, length = reals.shape[0], reals.shape[2]
    with torch.amp.autocast('cuda'):
        conditioning = diffusion.conditioner(metadata, device)
    
    video_exist = torch.stack([item['video_exist'] for item in metadata],dim=0)
    conditioning['metaclip_features'][~video_exist] = diffusion.model.model.empty_clip_feat
    conditioning['sync_features'][~video_exist] = diffusion.model.model.empty_sync_feat

    cond_inputs = diffusion.get_conditioning_inputs(conditioning)
    if batch_size > 1:
        noise_list = []
        for _ in range(batch_size):
            noise_1 = torch.randn([1, diffusion.io_channels, length]).to(device)  # 每次生成推进RNG状态
            noise_list.append(noise_1)
        noise = torch.cat(noise_list, dim=0)
    else:
        noise = torch.randn([batch_size, diffusion.io_channels, length]).to(device)

    with torch.amp.autocast('cuda'):

        model = diffusion.model
        if diffusion_objective == "v":
            fakes = sample(model, noise, 24, 0, **cond_inputs, cfg_scale=5, batch_cfg=True)
        elif diffusion_objective == "rectified_flow":
            fakes = sample_discrete_euler(model, noise, 24, **cond_inputs, cfg_scale=5, batch_cfg=True)
        if diffusion.pretransform is not None:
            fakes = diffusion.pretransform.decode(fakes)

    audios = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    return audios


def main():
    args = get_all_args()

    if args.save_dir == '':
        args.save_dir = args.results_dir


    seed = args.seed
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))
    seed_everything(seed, workers=True)

    # Load config
    if args.model_config == '':
        args.model_config = "PrismAudio/configs/model_configs/thinksound.json"
    with open(args.model_config) as f:
        model_config = json.load(f)

    duration = float(args.duration_sec)
    sample_rate = model_config["sample_rate"]
    latent_length = round(44100 / 64 / 32 * duration)

    model_config["sample_size"] = duration * sample_rate
    model_config["model"]["diffusion"]["config"]["sync_seq_len"] = 24 * int(duration)
    model_config["model"]["diffusion"]["config"]["clip_seq_len"] = 8 * int(duration)
    model_config["model"]["diffusion"]["config"]["latent_seq_len"] = latent_length

    model = create_model_from_config(model_config)
    if args.compile:
        model = torch.compile(model)

    model.load_state_dict(torch.load(args.ckpt_dir))
    vae_state = load_ckpt_state_dict(args.pretransform_ckpt_path, prefix='autoencoder.')
    model.pretransform.load_state_dict(vae_state)


    if args.dataset_config == '':
        args.dataset_config = "PrismAudio/configs/multimodal_dataset_demo.json"
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
        
    for td in dataset_config["test_datasets"]:
        td["path"] = args.results_dir

    dm = DataModule(
        dataset_config, 
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=(float)(args.duration_sec) * model_config["sample_rate"],
        audio_channels=model_config.get("audio_channels", 2),
        latent_length=round(44100/64/32*duration),
    )
    dm.setup('predict')
    dl = dm.predict_dataloader()

    current_date = datetime.now()
    formatted_date = current_date.strftime('%m%d')
    
    audio_dir = os.path.join(args.save_dir,f'{formatted_date}_batch_size'+str(args.test_batch_size))
    os.makedirs(audio_dir,exist_ok=True)

    for batch in tqdm(dl, desc="Predicting"):
        audio = predict_step(
            model,
            batch=batch,
            diffusion_objective=model_config["model"]["diffusion"]["diffusion_objective"],
            device='cuda:0'
        )

        _, metadata = batch
        ids = [item['id'] for item in metadata]

        for i in range(audio.size(0)):
            id_str = ids[i] if i < len(ids) else f"unknown_{i}"
            torchaudio.save(os.path.join(audio_dir, f"{id_str}.wav"), audio[i], 44100)

if __name__ == '__main__':
    main()