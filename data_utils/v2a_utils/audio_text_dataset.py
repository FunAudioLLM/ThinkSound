import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from torchvision.utils import save_image
from transformers import AutoProcessor
import torch.nn.functional as F
import numpy as np

import logging
log = logging.getLogger()

_CLIP_SIZE = 224
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class Audio_Text(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        *,
        tsv_path: Union[str, Path] = 'dataset/vggsound/split_txt/train_caption.csv',
        sample_rate: int = 44_100,
        duration_sec: float = 9.0,
        audio_samples: Optional[int] = 397312,
        normalize_audio: bool = False,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        save_dir: str = 'data/vggsound/video_latents_text/train'
    ):
        self.root = Path(root)
        self.normalize_audio = normalize_audio
        if audio_samples is None:
            self.audio_samples = int(sample_rate * duration_sec)
        else:
            self.audio_samples = audio_samples
            effective_duration = audio_samples / sample_rate
            # make sure the duration is close enough, within 15ms
            assert abs(effective_duration - duration_sec) < 0.015, \
                f'audio_samples {audio_samples} does not match duration_sec {duration_sec}'

        # videos = sorted(os.listdir(self.root))
        # videos = set([Path(v).stem for v in videos])  # remove extensions
        videos = []
        self.labels = []
        self.videos = []
        self.cots = []
        missing_videos = []
        # read the tsv for subset information
        df_list = pd.read_csv(tsv_path, sep=',', dtype={'id': str}).to_dict('records')
        
        # 控制处理的行范围
        if start_row is not None and end_row is not None:
            df_list = df_list[start_row:end_row]
        for record in df_list:
            id = record['id']
            if os.path.exists(f'{save_dir}/{id}.pth'): continue
            label = record['caption']
            # if id in videos:
            self.labels.append(label)
            # print(label,'debug1!!!!!!!!!')
            self.cots.append(record['caption_cot'])
            # self.labels[id] = label
            self.videos.append(id)
            # else:
            #     missing_videos.append(id)

        log.info(f'{len(videos)} videos found in {root}')
        log.info(f'{len(self.videos)} videos found in {tsv_path}')
        log.info(f'{len(missing_videos)} videos missing in {root}')

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.expected_audio_length = self.audio_samples
        self.resampler = {}

    def sample(self, idx: int):
        video_id = self.videos[idx]
        label = self.labels[idx]
        cot = self.cots[idx]
        audio_path = os.path.join(self.root, f'{video_id}.wav')
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.root, f'{video_id}.flac')
            if not os.path.exists(audio_path):
                raise RuntimeError(f'Audio is not exist {audio_path}')
        audio_chunk, sample_rate = torchaudio.load(audio_path)
        if len(audio_chunk.shape) != 2:
            raise RuntimeError(f'error audio shape {video_id}')

        abs_max = audio_chunk[0].abs().max()

        if abs_max <= 1e-6:
            if audio_chunk.shape[0] > 1 and audio_chunk[1].abs().max() > 1e-6:
                audio_chunk = audio_chunk[1:2]
            else:
                raise RuntimeError(f'Audio is silent {video_id}')

        # ensure the stereo audio
        if audio_chunk.shape[0] < 2:
            audio_chunk = audio_chunk.repeat(2, 1)
        elif audio_chunk.shape[0] > 2:
            audio_chunk = audio_chunk[:2]

        # resample
        if sample_rate == self.sample_rate:
            audio_chunk = audio_chunk
        else:
            if sample_rate not in self.resampler:
                # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method='sinc_interp_kaiser',
                    beta=14.769656459379492,
                )
            audio_chunk = self.resampler[sample_rate](audio_chunk)
        
        if audio_chunk.shape[1] < self.expected_audio_length:
            # zero-padding audio
            padding_length = self.expected_audio_length - audio_chunk.shape[1]
            # 创建 padding 张量，大小为 [batch_size, padding_length]，值为0
            padding = torch.zeros(audio_chunk.shape[0], padding_length)
            # 将原始音频和 padding 沿第 1 维度拼接在一起
            audio_chunk = torch.cat((audio_chunk, padding), dim=1)
            # raise RuntimeError(f'Audio too short {video_id}')
        audio_chunk = audio_chunk[:,:self.expected_audio_length]
        assert audio_chunk.shape == (2, 397312), f'error shape:{video_id},{audio_chunk.shape}'
        # print(label,'debug2!!!!!!!!!')
        data = {
            'id': video_id,
            'caption': label,
            'caption_cot': cot,
            'audio': audio_chunk,
        }

        return data

    def __getitem__(self, idx: int):
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.videos[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.labels)


# dataset = VGGSound(
#         root="data/vggsound/video/train",
#         tsv_path="data/vggsound/split_txt/temp.csv",
#         sample_rate=44100,
#         duration_sec=9.0,
#         audio_samples=397312,
#         start_row=0,
#         end_row=None,
#         save_dir="data/vggsound/video_224_latents_text/train"
#     )
# dataset[0]