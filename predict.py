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
from PrismAudio.models import create_model_from_config
from PrismAudio.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from PrismAudio.inference.sampling import sample, sample_discrete_euler
from pathlib import Path



def predict_step(diffusion, batch, diffusion_objective, device='cuda:0'):
    diffusion = diffusion.to(device)

    reals, metadata = batch
    ids = [item['id'] for item in metadata]
    batch_size, length = reals.shape[0], reals.shape[2]
    print(f"Predicting {batch_size} samples with length {length} for ids: {ids}")
    with torch.amp.autocast('cuda'):
        conditioning = diffusion.conditioner(metadata, device)
    
    video_exist = torch.stack([item['video_exist'] for item in metadata],dim=0)
    if 'metaclip_features' in conditioning:
        conditioning['metaclip_features'][~video_exist] = diffusion.model.model.empty_clip_feat

    if 'sync_features' in conditioning:
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
            import time
            start_time = time.time()
            fakes = sample_discrete_euler(model, noise, 24, **cond_inputs, cfg_scale=5, batch_cfg=True)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"执行时间: {execution_time:.2f} 秒")
        if diffusion.pretransform is not None:
            fakes = diffusion.pretransform.decode(fakes)

    audios = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    return audios



def load_file(filename, info, latent_length):
    # try:
    npz_file = filename
    if os.path.exists(npz_file): 
        # print(filename)
        npz_data = np.load(npz_file,allow_pickle=True)
        data = {key: npz_data[key] for key in npz_data.files}
        # print("data.keys()",data.keys())
        for key in data.keys():
            if isinstance(data[key], np.ndarray) and np.issubdtype(data[key].dtype, np.number):
                data[key] = torch.from_numpy(data[key])
    else:
        raise ValueError(f'error load file: {filename}')
    info.update(data)
    audio = torch.zeros((1, 64, latent_length), dtype=torch.float32)
    info['video_exist'] = torch.tensor(True)
    # except:
    #     print(f'error load file: {filename}')
    return audio, info

def load(filename,duration):
    assert os.path.exists(filename)
    info = {}
    audio, info = load_file(filename, info, round(44100/64/32*duration))
    info["path"] = filename

    info['id'] = Path(filename).stem
    info["relpath"] = 'demo.npz'

    return (audio, info)

def main():

    args = get_all_args()

    if (args.save_dir == ''):
        args.save_dir=args.results_dir


    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    # random.seed(seed)
    # torch.manual_seed(seed)
    seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    if args.model_config == '':
        args.model_config = "PrismAudio/configs/model_configs/thinksound.json"
    with open(args.model_config) as f:
        model_config = json.load(f)


    duration=(float)(args.duration_sec)
    
    model_config["sample_size"] = duration * model_config["sample_rate"]
    if "sync_seq_len" in model_config["model"]["diffusion"]["config"]:
        model_config["model"]["diffusion"]["config"]["sync_seq_len"] = 24 * int(duration)

    if "clip_seq_len" in model_config["model"]["diffusion"]["config"]:
        model_config["model"]["diffusion"]["config"]["clip_seq_len"] = 8 * int(duration)

    if "latent_seq_len" in model_config["model"]["diffusion"]["config"]:
        model_config["model"]["diffusion"]["config"]["latent_seq_len"] = round(44100 / 64 / 32 * duration)


    model = create_model_from_config(model_config)

    ## speed by torch.compile
    if args.compile:
        model = torch.compile(model)
        

    model.load_state_dict(torch.load(args.ckpt_dir))


    load_vae_state = load_ckpt_state_dict(args.pretransform_ckpt_path, prefix='autoencoder.') 
    model.pretransform.load_state_dict(load_vae_state)

    audio,meta=load(os.path.join(args.results_dir, "demo.npz") , duration)
    
    for k, v in meta.items():
        if isinstance(v, torch.Tensor):
            meta[k] = v.to('cuda:0')

    audio=predict_step(model, 
        batch=[audio,(meta,)],
        diffusion_objective=model_config["model"]["diffusion"]["diffusion_objective"], 
        device='cuda:0'
    )

    current_date = datetime.now()
    formatted_date = current_date.strftime('%m%d')
    
    audio_dir = os.path.join(args.save_dir,f'{formatted_date}_batch_size'+str(args.test_batch_size))
    os.makedirs(audio_dir,exist_ok=True)
    torchaudio.save(os.path.join(audio_dir,"demo.wav"), audio[0], 44100)
    

    #trainer.predict(training_wrapper, dm, return_predictions=False)

    

if __name__ == '__main__':
    main()