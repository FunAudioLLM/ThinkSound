import logging
log = logging.getLogger()
import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image
from transformers import AutoProcessor
import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
import mediapy
import torch.nn.functional as F
import numpy as np
import subprocess
from torchvision.utils import save_image
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

_CLIP_FPS = 4
_CLIP_SIZE = 288
_SYNC_FPS = 25
_SYNC_SIZE = 224

def pad_to_square(video_tensor):
    if len(video_tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (l, c, h, w)")

    l, c, h, w = video_tensor.shape
    max_side = max(h, w)

    pad_h = max_side - h
    pad_w = max_side - w
    
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    video_padded = F.pad(video_tensor, pad=padding, mode='constant', value=0)

    video_tensor = F.interpolate(video_padded, size=(_CLIP_SIZE, _CLIP_SIZE), mode='bilinear', align_corners=False)
    return video_tensor


def get_video_duration(video_path):
    video = VideoFileClip(str(video_path))
    return video.duration


class VGGSound(Dataset):
    def __init__(
        self,
        root: Path,
        *,
        tsv_path: Path,
        sample_rate: int = 44100,
        normalize_audio: bool = False,
        start_row: int = None,
        end_row: int = None,
        save_dir: str = '',
        use_variable_length: bool = False,
        video_encoder: str = 'videoprism',
        video_resolution: int = _CLIP_SIZE,
        inference_mode: bool = False,
        video_fps: int = _CLIP_FPS
    ):
        self.inference_mode = inference_mode
        self.sample_rate=sample_rate
        self.root = Path(root)
        self.normalize_audio = normalize_audio
        self.use_variable_length = use_variable_length
        self.video_encoder = video_encoder
        self.video_resolution = video_resolution
        self.video_fps = video_fps
        

        self.videos = []
        self.caption_cot = []
        df_list = pd.read_csv(tsv_path, sep=',', dtype={'id': str}).to_dict('records')
        if start_row is not None and end_row is not None:
            df_list = df_list[start_row:end_row]
        
        for record in df_list:
            id = record['id']
            if os.path.exists(f'{save_dir}/{id}.npz'): continue

            caption_cot = record['caption_cot']
 
            if not os.path.exists(os.path.join(self.root, id)+".mp4"):
                continue

            self.videos.append(id)
            self.caption_cot.append(caption_cot)


        log.info(f'processing {len(self.videos)} videos')


        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


        self.resampler = {}

    def sample(self, idx: int) -> dict[str, torch.Tensor]:

        video_id = self.videos[idx]
        caption_cot = self.caption_cot[idx]
        duration_sec= get_video_duration(self.root / (video_id + '.mp4'))
        reader = StreamingMediaDecoder(self.root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        ) 
        if not self.inference_mode:
            reader.add_basic_audio_stream(frames_per_chunk=2**30,)

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]

        sync_chunk = data_chunk[1]
        if not self.inference_mode:
            audio_chunk = data_chunk[2]
            audio_chunk = audio_chunk.transpose(0, 1)
        else:
            num_samples = int(self.sample_rate * duration_sec)
            audio_chunk = torch.randn((2, num_samples))

        if len(audio_chunk.shape) != 2:
            raise RuntimeError(f'error audio shape {video_id}')
        if clip_chunk is None:
            raise RuntimeError(f'CLIP video returned None {video_id}')


        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')
        if not self.inference_mode:
            sample_rate = int(reader.get_out_stream_info(2).sample_rate)
        else:
            sample_rate = self.sample_rate

        
        abs_max = audio_chunk[0].abs().max()

        if self.normalize_audio:
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95

        clip_expected_length = int(_CLIP_FPS * duration_sec)
        sync_expected_length = int(_SYNC_FPS * duration_sec)

        clip_chunk = clip_chunk[:clip_expected_length]

        if clip_chunk.shape[0] != clip_expected_length:
            current_length = clip_chunk.shape[0]
            padding_needed = clip_expected_length - current_length
            
            # If assertion passes, proceed with padding
            if padding_needed > 0:
                last_frame = clip_chunk[-1]
                padding = last_frame.repeat(padding_needed, 1, 1, 1)
                clip_chunk = torch.cat((clip_chunk, padding), dim=0)

        

        clip_chunk = pad_to_square(clip_chunk)
        clip_chunk = clip_chunk.permute(0, 2, 3, 1)
        
        clip_chunk = mediapy.to_float01(clip_chunk)

        sync_chunk = sync_chunk[:sync_expected_length]
        if sync_chunk.shape[0] != sync_expected_length:
            # padding using the last frame, but no more than 2
            current_length = sync_chunk.shape[0]
            last_frame = sync_chunk[-1]

            padding = last_frame.repeat(sync_expected_length - current_length, 1, 1, 1)
            sync_chunk = torch.cat((sync_chunk, padding), dim=0)

        
        sync_chunk = self.sync_transform(sync_chunk)
        
        data = {
            'id': video_id,
            'caption_cot': caption_cot,
            'audio': audio_chunk,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
        }

        return data



    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            logging.error(f'Error loading {self.videos[idx]}: {e}')
            return None

    def __len__(self) -> int:
        return len(self.videos)



