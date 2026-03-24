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

def save_tensor_as_image(tensor, save_path):
    """
    将形状为 (1, 3, H, W) 的 RGB 图像数组保存为图片文件。

    :param tensor: 输入的 NumPy 数组 (1, 3, H, W)。
    :param save_path: 图片保存路径。
    """
    # # 移除批次维度，变成 (3, H, W)
    # tensor = tensor.squeeze(0)
    
    # 交换轴顺序，变为 (H, W, 3)
    image_array = np.transpose(tensor, (1, 2, 0))
    
    # 检查数组是否为合适的数据类型
    if image_array.dtype != np.uint8:
        # 如果不是 uint8，首先标准化，然后转换
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        image_array = image_array.astype(np.uint8)
    
    # 创建图像对象
    image = Image.fromarray(image_array)
    
    # 保存图片
    image.save(save_path)
    print(f"Image saved to {save_path}")

def pad_to_square(video_tensor):
    # 验证输入的形状
    if len(video_tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (l, c, h, w)")

    l, c, h, w = video_tensor.shape
    max_side = max(h, w)

    # 计算每一维度需要的填充量：(left, right, top, bottom)
    pad_h = max_side - h
    pad_w = max_side - w
    
    # 创建padding tuple (left, right, top, bottom)
    # 因为图像的填充是作用在最后两个维度 h 和 w 上，所以我们需要指定这两个维度的填充
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    # 使用F.pad对视频张量进行填充操作
    # 填充参数为 (left, right, top, bottom)
    video_padded = F.pad(video_tensor, pad=padding, mode='constant', value=0)

    return video_padded

class VGGSound(Dataset):

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
        self.caption_cot = []
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
            caption_cot = record['caption_cot']
            # if id in videos:
            self.labels.append(label)
            # self.labels[id] = label
            self.videos.append(id)
            self.caption_cot.append(caption_cot)
            # else:
            #     missing_videos.append(id)

        log.info(f'{len(videos)} videos found in {root}')
        log.info(f'{len(self.videos)} videos found in {tsv_path}')
        log.info(f'{len(missing_videos)} videos missing in {root}')

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.expected_audio_length = self.audio_samples
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose([
            v2.Lambda(pad_to_square),          # 先填充为正方形
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.clip_processor = AutoProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
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
        label = self.labels[idx]
        caption_cot = self.caption_cot[idx]

        reader = StreamingMediaDecoder(self.root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        ) 
        # reader.add_basic_audio_stream(frames_per_chunk=2**30,)

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
        # audio_chunk = data_chunk[2]
        # if len(audio_chunk.shape) != 2:
        #     raise RuntimeError(f'error audio shape {video_id}')
        if clip_chunk is None:
            raise RuntimeError(f'CLIP video returned None {video_id}')
        # if clip_chunk.shape[0] < self.clip_expected_length:
        #     raise RuntimeError(
        #         f'CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}'
        #     )

        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')

        # truncate the video
        clip_chunk = clip_chunk[:self.clip_expected_length]
        # import ipdb
        # ipdb.set_trace()
        if clip_chunk.shape[0] != self.clip_expected_length:
            current_length = clip_chunk.shape[0]
            padding_needed = self.clip_expected_length - current_length
            
            # Check that padding needed is no more than 2
            # assert padding_needed < 4, f'Padding no more than 2 frames allowed, but {padding_needed} needed'

            # If assertion passes, proceed with padding
            if padding_needed > 0:
                last_frame = clip_chunk[-1]
                log.info(last_frame.shape) 
                # Repeat the last frame to reach the expected length
                padding = last_frame.repeat(padding_needed, 1, 1, 1)
                clip_chunk = torch.cat((clip_chunk, padding), dim=0)
            # raise RuntimeError(f'CLIP video wrong length {video_id}, '
            #                    f'expected {self.clip_expected_length}, '
            #                    f'got {clip_chunk.shape[0]}')
        
        # save_image(clip_chunk[0] / 255.0,'ori.png')
        clip_chunk = pad_to_square(clip_chunk)
        # save_image(clip_chunk[0] / 255.0,'square.png')
        # clip_chunk = self.clip_transform(clip_chunk)
        # import ipdb
        # ipdb.set_trace()
        clip_chunk = self.clip_processor(images=clip_chunk, return_tensors="pt")["pixel_values"]
        # log.info(clip_chunk.shape)
        # save_tensor_as_image(clip_chunk[0].numpy(),'scale.png')
        # log.info(clip_chunk[0])
        # clip_chunk = outputs
        # text_ids = outputs["input_ids"]
        # temp_img = clip_chunk[0].permute(1, 2, 0) * 255
        # save_image(clip_chunk[0],'scale.png')
        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            # padding using the last frame, but no more than 2
            current_length = sync_chunk.shape[0]
            last_frame = sync_chunk[-1]
            # 重复最后一帧以进行填充
            padding = last_frame.repeat(self.sync_expected_length - current_length, 1, 1, 1)
            # assert self.sync_expected_length - current_length < 12, f'sync can pad no more than 2 while {self.sync_expected_length - current_length}'
            sync_chunk = torch.cat((sync_chunk, padding), dim=0)
            # raise RuntimeError(f'Sync video wrong length {video_id}, '
            #                    f'expected {self.sync_expected_length}, '
            #                    f'got {sync_chunk.shape[0]}')
        
        sync_chunk = self.sync_transform(sync_chunk)
        # assert audio_chunk.shape[1] == self.expected_audio_length and clip_chunk.shape[0] == self.clip_expected_length \
        # and sync_chunk.shape[0] == self.sync_expected_length, 'error processed data shape'
        data = {
            'id': video_id,
            'caption': label,
            # 'audio': audio_chunk,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
            'caption_cot': caption_cot,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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