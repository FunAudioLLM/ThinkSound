import math
import random
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
import numpy as np

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor, randomize=True) -> Tuple[torch.Tensor, float, float, int, int]:

        n_channels, n_samples = source.shape
        
        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(randomize and n_samples > self.n_samples):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PadCrop_Video_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, fps: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.fps = fps
        self.n_frames = int(self.fps * self.n_samples / self.sample_rate)

    def __call__(self, audio: torch.Tensor, video: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        n_channels, n_samples = audio.shape
        # print(video.shape)
        n_frames, dim = video.shape
        if not torch.is_tensor(video):
            video = torch.from_numpy(video)
        # If the audio is shorter than the desired length, pad it
        audio_upper_bound = max(0, n_samples - self.n_samples)
        video_upper_bound = int(max(0, n_frames - self.n_frames) * self.sample_rate / self.fps)
        upper_bound = min(audio_upper_bound,video_upper_bound)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples and n_frames > self.n_frames):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)
        frame_offset = int(self.fps * offset / self.sample_rate)
        # frame_end = frame_offset + int(self.fps * self.n_samples / self.sample_rate)
        # Create the chunk
        chunk = audio.new_zeros([n_channels, self.n_samples])
        video_chunk = video.new_zeros([self.n_frames, video.shape[1]])
        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = audio[:, offset:offset + self.n_samples]
        video_chunk[:min(n_frames, self.n_frames)] = video[frame_offset:frame_offset + self.n_frames,:]
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            video_chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PadCrop_Video_Image_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, fps: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.fps = fps
        self.n_frames = int(self.fps * self.n_samples / self.sample_rate)

    def __call__(self, audio: torch.Tensor, video: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        n_channels, n_samples = audio.shape
        # import ipdb
        # ipdb.set_trace()
        n_frames, channel, width, height= video.shape
        video = torch.from_numpy(video)
        # If the audio is shorter than the desired length, pad it
        audio_upper_bound = max(0, n_samples - self.n_samples)
        video_upper_bound = int(max(0, n_frames - self.n_frames) * self.sample_rate / self.fps)
        upper_bound = min(audio_upper_bound,video_upper_bound)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples and n_frames > self.n_frames):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)
        frame_offset = int(self.fps * offset / self.sample_rate)
        # frame_end = frame_offset + int(self.fps * self.n_samples / self.sample_rate)
        # Create the chunk
        chunk = audio.new_zeros([n_channels, self.n_samples])
        video_chunk = video.new_zeros([self.n_frames, channel, width, height])
        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = audio[:, offset:offset + self.n_samples]
        video_chunk[:min(n_frames, self.n_frames)] = video[frame_offset:frame_offset + self.n_frames]
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            video_chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PadCrop_Video_Hiera_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, fps: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.fps = fps
        self.n_frames = int(self.fps * self.n_samples / self.sample_rate)

    def __call__(self, audio: torch.Tensor, video: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:

        n_channels, n_samples = audio.shape
        n_frames, heigh, width, channel = video.shape
        video = torch.from_numpy(video)
        # If the audio is shorter than the desired length, pad it
        audio_upper_bound = max(0, n_samples - self.n_samples)
        video_upper_bound = int(max(0, n_frames - self.n_frames) * self.sample_rate / self.fps)
        upper_bound = min(audio_upper_bound,video_upper_bound)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples and n_frames > self.n_frames):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)
        frame_offset = int(self.fps * offset / self.sample_rate)
        # frame_end = frame_offset + int(self.fps * self.n_samples / self.sample_rate)
        # Create the chunk
        chunk = audio.new_zeros([n_channels, self.n_samples])
        video_chunk = video.new_zeros([self.n_frames, heigh, width, channel])
        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = audio[:, offset:offset + self.n_samples]
        video_chunk[:min(n_frames, self.n_frames)] = video[frame_offset:frame_offset + self.n_frames]
        # video_chunk = video_chunk[None].permute(0, 4, 1, 2, 3).contiguous()
        # print(video_chunk.shape)
        # video_chunk = F.interpolate(
        #     video_chunk[0],
        #     size=(224, 224, 3),  # 输出的空间尺寸
        #     scale_factor=(target_frames / video_tensor.shape[1], 1, 1),  # 时间轴的缩放因子
        #     mode='trilinear',  # 使用三线性插值
        #     align_corners=False
        # )

        # video_chunk = F.interpolate(video_chunk, size=(64, 224, 224), mode="trilinear")[0]
        # video_chunk = video_chunk.view(3,4,16,224,224).transpose(0,1)
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            video_chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PadCrop_DualVideo_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, fps: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.fps = fps
        self.n_frames = int(self.fps * self.n_samples / self.sample_rate)

    def __call__(self, audio: torch.Tensor, video_360: torch.Tensor, video_fov: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        n_channels, n_samples = audio.shape
        # print(video.shape)
        n_frames, dim = video_360.shape
        video_360 = torch.from_numpy(video_360)
        video_fov = torch.from_numpy(video_fov)
        # If the audio is shorter than the desired length, pad it
        audio_upper_bound = max(0, n_samples - self.n_samples)
        video_upper_bound = int(max(0, n_frames - self.n_frames) * self.sample_rate / self.fps)
        upper_bound = min(audio_upper_bound,video_upper_bound)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples and n_frames > self.n_frames):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)
        frame_offset = int(self.fps * offset / self.sample_rate)
        # frame_end = frame_offset + int(self.fps * self.n_samples / self.sample_rate)
        # Create the chunk
        chunk = audio.new_zeros([n_channels, self.n_samples])
        video_360_chunk = video_360.new_zeros([self.n_frames, video_360.shape[1]])
        video_fov_chunk = video_fov.new_zeros([self.n_frames, video_fov.shape[1]])
        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = audio[:, offset:offset + self.n_samples]
        video_360_chunk[:min(n_frames, self.n_frames)] = video_360[frame_offset:frame_offset + self.n_frames,:]
        video_fov_chunk[:min(n_frames, self.n_frames)] = video_fov[frame_offset:frame_offset + self.n_frames,:]
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            video_360_chunk,
            video_fov_chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

class FOA(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1:  # s -> (4, s)
        foa = torch.zeros(4, signal_shape[0], device=signal.device)  # 与输入信号一致的设备类型
        foa[0, :] = signal  # W通道: 全方位声源
        foa[1, :] = 0  # X通道
        foa[2, :] = 0  # Y通道
        foa[3, :] = 0  # Z通道
    elif len(signal_shape) == 2:
        foa = torch.zeros(4, signal_shape[1], device=signal.device)  # 与输入信号一致的设备类型
        if signal_shape[0] == 1:  # (1, s) -> (4, s)
            foa[0, :] = signal[0]  # W通道: 全方位声源
            foa[1, :] = 0  # X通道
            foa[2, :] = 0  # Y通道
            foa[3, :] = 0  # Z通道
        elif signal_shape[0] == 2:  # (2, s) -> (4, s)
            left = signal[0]
            right = signal[1]
            # 将立体声信号映射到FOA信号通道
            foa[0, :] = (left + right) / np.sqrt(2)  # W通道: 全方位声源
            foa[1,  :] = (left - right) / np.sqrt(2)  # X通道: 前后方向
            foa[2, :] = 0  # Y通道: 左右方向，简单实现先置零
            foa[3, :] = 0  # Z通道: 垂直方向，这里置零
        else:
            foa = signal

    else:
        raise ValueError(f"Unsupported signal shape: {signal_shape}")

    assert foa.shape[0] == 4, f'inputs not FOA format'

    return foa