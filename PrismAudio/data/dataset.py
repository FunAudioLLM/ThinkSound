import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import subprocess
import time
import torch
import torchaudio
import webdataset as wds
import pandas as pd
from aeiou.core import is_silence
from os import path
from pathlib import Path
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from typing import Optional, Callable, List
import bisect

from .utils import FOA, Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, PadCrop_Video_Normalized_T, PadCrop_Video_Hiera_Normalized_T, PadCrop_Video_Image_Normalized_T, PadCrop_DualVideo_Normalized_T

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames





class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        split_path: str,
        audio_dir: str = None,
        custom_metadata_fn: Optional[Callable[[str], str]] = None
    ):
        self.id = id
        self.path = path
        self.split_path = split_path
        self.audio_dir = audio_dir
        self.custom_metadata_fn = custom_metadata_fn

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        input_type="prompt",
        fps=4,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []
        if input_type == 'video':
            self.pad_crop = PadCrop_Video_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
        elif input_type == 'video_hiera':
            self.pad_crop = PadCrop_Video_Hiera_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
        elif input_type == 'video_image':
            self.pad_crop = PadCrop_Video_Image_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
        elif input_type == 'dual_video':
            self.pad_crop = PadCrop_DualVideo_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
        else:
            self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels
        print('######################')
        print(f'input channels is: {force_channels}')
        print('######################')
        self.encoding = torch.nn.Sequential(
            FOA() if self.force_channels == "foa" else torch.nn.Identity(),
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.input_type = input_type
        self.sr = sample_rate
        self.custom_metadata_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            def add_prefix(s):
                return str(os.path.join(config.path,f'{s.strip()}'))
            with open(config.split_path,'r') as f:
                item_names = f.readlines()
            filenames = list(map(add_prefix, item_names))
            self.filenames.extend(filenames) 
            # self.filenames.extend(get_audio_filenames(config.path, keywords))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename):
        ext = filename.split(".")[-1]
        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            try:
                resample_tf = T.Resample(in_sr, self.sr)
                audio = resample_tf(audio)
            except:
                print(f'{filename} resample errors')

        assert not (torch.isnan(audio).any() or torch.isinf(audio).any()), f'file-{filename} contains nan or inf number, check it!'
        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        assert os.path.exists(audio_filename), f'{audio_filename}: file not exists'
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)
            info = {}
            info["path"] = audio_filename

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)


            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]
            if self.input_type == 'video':
                audio, video, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio, info['video'])
                info['video'] = video
            elif self.input_type == 'dual_video':
                audio, video_360, video_fov, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio, info['video'], info['video_fov'])
                info['video_360'] = video_360
                info['video_fov'] = video_fov
            else:
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
                assert not (torch.isnan(audio).any() or torch.isinf(audio).any()), f'file-{filename} contains nan or inf number, check it!'
            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)


            
            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask

            end_time = time.time()
            info["load_time"] = end_time - start_time
            
            
            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]

class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        input_type="prompt",
        fps=4,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        self.force_channels = force_channels
        print('######################')
        print(f'input channels is: {force_channels}')
        print('######################')
        self.encoding = torch.nn.Sequential(
            FOA() if self.force_channels == "foa" else torch.nn.Identity(),
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.input_type = input_type
        self.sr = sample_rate
        for config in configs:
            self.root_paths.append(config.path)
            def add_prefix(s):
                return str(os.path.join(config.path,f'{s.strip()}'))
            with open(config.split_path,'r') as f:
                item_names = f.readlines()
            filenames = list(map(add_prefix, item_names))
            self.filenames.extend(filenames) 
            # self.filenames.extend(get_audio_filenames(config.path, keywords))
            

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename, info):
        # try:
        npz_file = filename.replace('.pth','.npz')
        if os.path.exists(filename) and '.npz' not in filename:
            data = torch.load(filename, weights_only=False)
        elif os.path.exists(npz_file): 
            # print(filename)
            npz_data = np.load(npz_file,allow_pickle=True)
            data = {key: npz_data[key] for key in npz_data.files}
            # print("data.keys()",data.keys())
            for key in data.keys():
                if isinstance(data[key], np.ndarray) and np.issubdtype(data[key].dtype, np.number):
                    data[key] = torch.from_numpy(data[key])
        else:
            raise ValueError(f'error load file with file not exists: {filename}')
        info.update(data)
        audio = data['latent']
        # except:
        #     print(f'error load file: {filename}')
        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        assert os.path.exists(audio_filename) or audio_filename.replace('.pth','.npz'), f'{audio_filename}: file not exists'
        # try:
        start_time = time.time()
        info = {}
        audio = self.load_file(audio_filename, info)
        info["path"] = audio_filename
        info['id'] = Path(audio_filename).stem
        for root_path in self.root_paths:
            if root_path in audio_filename:
                info["relpath"] = path.relpath(audio_filename, root_path)
        
        return (audio, info)

class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        input_type="prompt",
        fps=4,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        self.force_channels = force_channels
        print('######################')
        print(f'input channels is: {force_channels}')
        print('######################')
        self.encoding = torch.nn.Sequential(
            FOA() if self.force_channels == "foa" else torch.nn.Identity(),
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.fake_clip_features = torch.zeros(72, 1024)
        self.fake_sync_features = torch.zeros(216, 768)
        self.video_exist = torch.tensor(0, dtype=torch.bool)
        self.input_type = input_type
        self.sr = sample_rate
        for config in configs:
            self.root_paths.append(config.path)
            def add_prefix(s):
                return str(os.path.join(config.path,f'{s.strip()}'))
            with open(config.split_path,'r') as f:
                item_names = f.readlines()
            filenames = list(map(add_prefix, item_names))
            self.filenames.extend(filenames) 
            # self.filenames.extend(get_audio_filenames(config.path, keywords))
            

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename, info):
        # try:
        npz_file = filename.replace('.pth','.npz')
        if os.path.exists(filename) and '.npz' not in filename:
            data = torch.load(filename, weights_only=False)
        elif os.path.exists(npz_file): 
            # print(filename)
            npz_data = np.load(npz_file,allow_pickle=True)
            data = dict(npz_data)
            # print("data.keys()",data.keys())
            for key in data.keys():
                if isinstance(data[key], np.ndarray) and np.issubdtype(data[key].dtype, np.number):
                    data[key] = torch.from_numpy(data[key])
        else:
            raise ValueError(f'error load file: {filename}')
        info.update(data)
        audio = data['latent']
        if 'source_latent' not in data.keys():
            info['source_latent']= audio
        info['video_features'] = self.fake_clip_features
        info['sync_features'] = self.fake_sync_features
        info['video_exist'] = self.video_exist
        # except:
        #     print(f'error load file: {filename}')
        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        assert os.path.exists(audio_filename) or audio_filename.replace('.pth','.npz'), f'{audio_filename}: file not exists'
        # try:
        start_time = time.time()
        info = {}
        audio = self.load_file(audio_filename, info)
        info["path"] = audio_filename
        assert audio.shape == (64,194), f'{audio.shape} input error, id: {audio_filename}'
        info['id'] = Path(audio_filename).stem
        for root_path in self.root_paths:
            if root_path in audio_filename:
                info["relpath"] = path.relpath(audio_filename, root_path)
        
        return (audio, info)




class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        input_type="prompt",
        fps=4,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []
        self.sample_size = sample_size
        self.force_channels = force_channels
        print('######################')
        print(f'input channels is: {force_channels}')
        print('######################')
        self.encoding = torch.nn.Sequential(
            FOA() if self.force_channels == "foa" else torch.nn.Identity(),
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.input_type = input_type
        self.sr = sample_rate
        self.video_exist = torch.tensor(1, dtype=torch.bool)
        self.audio_files = []
        for config in configs:
            self.root_paths.append(config.path)
            def add_prefix(s):
                return str(os.path.join(config.path,f'{s.strip()}'))
            with open(config.split_path,'r') as f:
                item_names = f.readlines()
            filenames = list(map(add_prefix, item_names))
            self.filenames.extend(filenames) 
            if config.audio_dir is not None:
                def add_prefix(s):
                    return str(os.path.join(config.audio_dir,f'{Path(s).stem}.wav'))
                filenames = list(map(add_prefix, item_names))
                self.audio_files.extend(filenames)
            # self.filenames.extend(get_audio_filenames(config.path, keywords))

        print(f'Found {len(self.filenames)} files')

    def load_audio(self, filename):
        ext = filename.split(".")[-1]
        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            try:
                resample_tf = T.Resample(in_sr, self.sr)
                audio = resample_tf(audio)
            except:
                print(f'{filename} resample errors')

        assert not (torch.isnan(audio).any() or torch.isinf(audio).any()), f'file-{filename} contains nan or inf number, check it!'
        return audio
    
    

    
    def check_audio_file(self, audio_path):
        # 首先检查原始路径是否存在
        if os.path.exists(audio_path):
            return audio_path
        # 如果不存在，尝试替换为.flac扩展名
        name, ext = os.path.splitext(audio_path)
        flac_path = f"{name}.flac"
        
        if os.path.exists(flac_path):
            return flac_path
        raise FileNotFoundError(f"音频文件不存在: {audio_path} 和 {flac_path} 都不存在")

    def load_file(self, filename, info):
        try:
            npz_file = filename.replace('.pth','.npz')
            if os.path.exists(filename) and '.npz' not in filename:
                data = torch.load(filename, weights_only=False)
            elif os.path.exists(npz_file): 
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
            audio = data['latent']
            info['video_exist'] = self.video_exist
        except Exception as e:
            print(f'error load file: {filename} with error: {e}')
            return None
        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
            loop = True
            while loop:
                filename = self.filenames[idx]
                if len(self.audio_files) > 0:
                    audio_path = self.audio_files[idx]
                    audio_path = self.check_audio_file(audio_path)
                    waveform = self.load_audio(audio_path)
                else:
                    waveform = None
                assert os.path.exists(filename) or filename.replace('.pth','.npz'), f'{filename}: file not exists'
                # try:
                start_time = time.time()
                info = {}
                audio = self.load_file(filename, info)
                if audio is not None:
                    loop = False
                else:
                    idx = (idx+1) % len(self.filenames)

            if waveform is not None:
                padded_waveform = torch.zeros(waveform.shape[0], self.sample_size, dtype=waveform.dtype)
                copy_length = min(waveform.shape[1], self.sample_size)
                padded_waveform[:, :copy_length] = waveform[:, :copy_length]

                waveform = padded_waveform
                waveform = waveform.clamp(-1, 1)
                # Encode the file to assist in prediction
                if self.encoding is not None:
                    waveform = self.encoding(waveform)
                info['waveform'] = waveform
            info["path"] = filename
            info['id'] = Path(filename).stem
            for root_path in self.root_paths:
                if root_path in filename:
                    info["relpath"] = path.relpath(filename, root_path)
            

            return (audio, info)

# modified from https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
class MultiModalDataset(torch.utils.data.Dataset):
    datasets: list[torch.utils.data.Dataset]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, video_datasets: list[torch.utils.data.Dataset], audio_datasets: list[torch.utils.data.Dataset]):
        super().__init__()
        self.video_datasets = list(video_datasets)
        self.audio_datasets = list(audio_datasets)
        self.datasets = self.video_datasets + self.audio_datasets

        self.cumulative_sizes = self.cumsum(self.datasets)
        print(f'Found {self.cumulative_sizes[-1]} files')

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video_datasets[0].compute_latent_stats()


# class MultiModalDataset(torch.utils.data.Dataset):
#     def __init__(
#         self, 
#         configs,
#         sample_size=65536, 
#         sample_rate=48000, 
#         keywords=None, 
#         random_crop=True,
#         input_type="prompt",
#         fps=4,
#         force_channels="stereo"
#     ):
#         super().__init__()
#         self.filenames = []
#         self.captions = []
#         self.caption_t5s = []
#         self.ids = []
#         self.augs = torch.nn.Sequential(
#             PhaseFlipper(),
#         )

#         self.root_paths = []
#         if input_type == 'video':
#             self.pad_crop = PadCrop_Video_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
#         elif input_type == 'video_hiera':
#             self.pad_crop = PadCrop_Video_Hiera_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
#         elif input_type == 'video_image':
#             self.pad_crop = PadCrop_Video_Image_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
#         elif input_type == 'dual_video':
#             self.pad_crop = PadCrop_DualVideo_Normalized_T(sample_size, sample_rate, fps, randomize=random_crop)
#         else:
#             self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

#         self.force_channels = force_channels
#         print('######################')
#         print(f'input channels is: {force_channels}')
#         print('######################')
#         self.encoding = torch.nn.Sequential(
#             FOA() if self.force_channels == "foa" else torch.nn.Identity(),
#             Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
#             Mono() if self.force_channels == "mono" else torch.nn.Identity(),
#         )
#         self.input_type = input_type
#         self.sr = sample_rate
#         self.custom_metadata_fns = {}

#         for config in configs:
#             print(config.split_path)
#             self.root_paths.append(config.path)
#             def add_prefix(s):
#                 return str(os.path.join(config.path,f'{s.strip()}'))
#             with open(config.split_path,'r') as f:
#                 item_names = f.readlines()
#             csv_path = config.split_path.replace('.txt','.csv')
#             df = pd.read_csv(csv_path)
#             # 检查是否存在 'caption_t5' 列，如果不存在则创建并复制 'caption' 的值
#             if 'caption_t5' not in df.columns:
#                 df['caption_t5'] = df['caption']

#             captions = df['caption'].tolist()
#             caption_t5s = df['caption_t5'].tolist()
#             filenames = list(map(add_prefix, item_names))
#             assert len(captions) == len(caption_t5s) and len(captions) == len(filenames), f'{config.path} has wrong filename and caption'
#             if config.id == 'vggsound':
#                 self.filenames.extend(filenames*5)
#                 self.captions.extend(captions*5)
#                 self.caption_t5s.extend(caption_t5s*5)
#                 self.ids.extend(df['id'].tolist()*5)
#             else:
#                 self.filenames.extend(filenames)
#                 self.captions.extend(captions)
#                 self.caption_t5s.extend(caption_t5s)
#                 self.ids.extend(df['id'].tolist())
#             # self.filenames.extend(get_audio_filenames(config.path, keywords))
#             if config.custom_metadata_fn is not None:
#                 self.custom_metadata_fns[config.path] = config.custom_metadata_fn

#         assert len(self.ids) == len(self.captions) and len(self.caption_t5s) == len(self.filenames), 'length need to be same'
#         print(f'Found {len(self.filenames)} files')


#     def load_file(self, filename):
#         ext = filename.split(".")[-1]
#         if ext == "mp3":
#             with AudioFile(filename) as f:
#                 audio = f.read(f.frames)
#                 audio = torch.from_numpy(audio)
#                 in_sr = f.samplerate
#         else:
#             audio, in_sr = torchaudio.load(filename, format=ext)

#         if in_sr != self.sr:
#             try:
#                 resample_tf = T.Resample(in_sr, self.sr)
#                 audio = resample_tf(audio)
#             except:
#                 print(f'{filename} resample errors')

#         assert not (torch.isnan(audio).any() or torch.isinf(audio).any()), f'file-{filename} contains nan or inf number, check it!'
#         return audio

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         audio_filename = self.filenames[idx]
#         id = self.ids[idx]
#         assert str(id) == str(Path(audio_filename).stem), f'audio_file: {audio_filename} needs to be same as {id} '
#         assert os.path.exists(audio_filename), f'{audio_filename}: file not exists'
#         try:
#             start_time = time.time()
#             audio = self.load_file(audio_filename)
#             caption = self.captions[idx]
#             caption_t5 = self.caption_t5s[idx]
#             if pd.isna(caption_t5) or caption_t5 == '':
#                 caption_t5 = caption
#             info = {}
#             info["path"] = audio_filename
#             info['caption'] = caption
#             info['caption_t5'] = caption_t5

#             for root_path in self.root_paths:
#                 if root_path in audio_filename:
#                     info["relpath"] = path.relpath(audio_filename, root_path)


#             for custom_md_path in self.custom_metadata_fns.keys():
#                 if custom_md_path in audio_filename:
#                     custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
#                     custom_metadata = custom_metadata_fn(info, audio)
#                     info.update(custom_metadata)

#                 if "__reject__" in info and info["__reject__"]:
#                     return self[random.randrange(len(self))]
#             # if self.input_type == 'video':
#             #     audio, video, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio, info['clip_features'])
#             #     info['clip_features'] = video
#             # else:
#             if info['flag']:
#                 audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio,randomize=False)
#             else:
#                 audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio,randomize=True)
#             assert not (torch.isnan(audio).any() or torch.isinf(audio).any()), f'file-{filename} contains nan or inf number, check it!'
#             # Run augmentations on this sample (including random crop)
#             if self.augs is not None:
#                 audio = self.augs(audio)

#             audio = audio.clamp(-1, 1)

#             # Encode the file to assist in prediction
#             if self.encoding is not None:
#                 audio = self.encoding(audio)


            
#             info["timestamps"] = (t_start, t_end)
#             info["seconds_start"] = seconds_start
#             info["seconds_total"] = seconds_total
#             info["padding_mask"] = padding_mask

#             end_time = time.time()
#             info["load_time"] = end_time - start_time
            
            
#             return (audio, info)
#         except Exception as e:
#             print(f'Couldn\'t load file {audio_filename}: {e}')
#             return self[random.randrange(len(self))]

def group_by_keys(data, keys=wds.tariterators.base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if wds.tariterators.trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if wds.tariterators.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            print(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if wds.tariterators.valid_sample(current_sample):
        yield current_sample

wds.tariterators.group_by_keys = group_by_keys

# S3 code and WDS preprocessing code based on implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile=None):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    # Construct the `aws s3 ls` command
    cmd = ['aws', 's3', 'ls', bucket_path]

    if profile is not None:
        cmd.extend(['--profile', profile])

    if recursive:
        # Add the --recursive flag if requested
        cmd.append('--recursive')
    
    # Run the `aws s3 ls` command and capture the output
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    # Split the output into lines and strip whitespace from each line
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    # Remove the timestamp from lines that begin with a timestamp
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x)
                if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    # Construct a full S3 path for each file in the contents list
    contents = [posixpath.join(s3_url_prefix or '', x)
                for x in contents if not x.endswith('/')]
    # Apply the filter, if specified
    if filter:
        contents = [x for x in contents if filter in x]
    # Remove redundant directory names in the S3 URL
    if recursive:
        # Get the main directory name from the S3 URL
        main_dir = "/".join(bucket_path.split('/')[3:])
        # Remove the redundant directory names from each file path
        contents = [x.replace(f'{main_dir}', '').replace(
            '//', '/') for x in contents]
    # Print debugging information, if requested
    if debug:
        print("contents = \n", contents)
    # Return the list of S3 paths to files
    return contents


def get_all_s3_urls(
    names=[],           # list of all valid [LAION AudioDataset] dataset names
    # list of subsets you want from those datasets, e.g. ['train','valid']
    subsets=[''],
    s3_url_prefix=None,  # prefix for those dataset names
    recursive=True,     # recursively list all tar files in all subdirs
    filter_str='tar',   # only grab files with this substring
    # print debugging info -- note: info displayed likely to change at dev's whims
    debug=False,
    profiles={},        # dictionary of profiles for each item in names, e.g. {'dataset1': 'profile1', 'dataset2': 'profile2'}
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        if s3_url_prefix is None:
            contents_str = name
        else:
            # Construct the S3 path using the s3_url_prefix and the current name value
            contents_str = posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            # Get the list of tar files in the current subset directory
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str, s3_url_prefix=None, recursive=recursive, filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                # Escape spaces and parentheses in the tar filename for use in the shell command
                tar = tar.replace(" ", "\ ").replace(
                    "(", "\(").replace(")", "\)")
                # Construct the S3 path to the current tar file
                s3_path = posixpath.join(name, subset, tar) + " -"
                # Construct the AWS CLI command to download the current tar file
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}"
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"
                if debug:
                    print("request_str = ", request_str)
                # Add the constructed URL to the list of URLs
                urls.append(request_str)
    return urls


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def is_valid_sample(sample):
    has_json = "json" in sample
    has_audio = "audio" in sample
    is_silent = is_silence(sample["audio"])
    is_rejected = "__reject__" in sample["json"] and sample["json"]["__reject__"]

    return has_json and has_audio and not is_silent and not is_rejected

class S3DatasetConfig:
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
        )

        return self.urls

class LocalWebDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):

        self.urls = fast_scandir(self.path, ["tar"])[1]

        return self.urls

def audio_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None

def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result

class WebDatasetDataLoader():
    def __init__(
        self,
        datasets: List[S3DatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        **data_loader_kwargs
    ):

        self.datasets = datasets

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase

        urls = [dataset.load_data_urls() for dataset in datasets]

        # Flatten the list of lists of URLs
        urls = [url for dataset_urls in urls for url in dataset_urls]

        # Shuffle the urls
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(audio_decoder, handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            #wds.shuffle(bufsize=1000, initial=5000),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        ).with_epoch(epoch_steps//num_workers if num_workers > 0 else epoch_steps)

        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, **data_loader_kwargs)

    def wds_preprocess(self, sample):

        found_key, rewrite_key = '', ''
        for k, v in sample.items():  # print the all entries in dict
            for akey in AUDIO_KEYS:
                if k.endswith(akey):
                    # to rename long/weird key with its simpler counterpart
                    found_key, rewrite_key = k, akey
                    break
            if '' != found_key:
                break
        if '' == found_key:  # got no audio!
            return None  # try returning None to tell WebDataset to skip this one

        audio, in_sr = sample[found_key]
        if in_sr != self.sample_rate:
            resample_tf = T.Resample(in_sr, self.sample_rate)
            audio = resample_tf(audio)

        if self.sample_size is not None:
            # Pad/crop and get the relative timestamp
            pad_crop = PadCrop_Normalized_T(
                self.sample_size, randomize=self.random_crop, sample_rate=self.sample_rate)
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(
                audio)
            sample["json"]["seconds_start"] = seconds_start
            sample["json"]["seconds_total"] = seconds_total
            sample["json"]["padding_mask"] = padding_mask
        else:
            t_start, t_end = 0, 1

        # Check if audio is length zero, initialize to a single zero if so
        if audio.shape[-1] == 0:
            audio = torch.zeros(1, 1)

        # Make the audio stereo and augment by randomly inverting phase
        augs = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
            PhaseFlipper() if self.augment_phase else torch.nn.Identity()
        )

        audio = augs(audio)

        sample["json"]["timestamps"] = (t_start, t_end)

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        # Check for custom metadata functions
        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue
        
            if dataset.path in sample["__url__"]:
                custom_metadata = dataset.custom_metadata_fn(sample["json"], audio)
                sample["json"].update(custom_metadata)

        if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
            del sample[found_key]

        sample["audio"] = audio

        # Add audio to the metadata as well for conditioning
        sample["json"]["audio"] = audio
        
        return sample

def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4, shuffle=True):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"
    
    if audio_channels == 1:
        force_channels = "mono"
    elif audio_channels == 2:
        force_channels = "stereo"
    else:
        force_channels = "foa"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            split_path = audio_dir_config.get("split_path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    split_path=split_path,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        train_set = SampleDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            input_type=dataset_config.get("input_type", "video"),
            fps=dataset_config.get("fps", 4),
            force_channels=force_channels
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)

    elif dataset_type in ["s3", "wds"]: # Support "s3" type for backwards compatibility

        wds_configs = []

        for wds_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            if "s3_path" in wds_config:

                wds_configs.append(
                    S3DatasetConfig(
                        id=wds_config["id"],
                        s3_path=wds_config["s3_path"],
                        custom_metadata_fn=custom_metadata_fn,
                        profile=wds_config.get("profile", None),
                    )
                )
            
            elif "path" in wds_config:
                    
                    wds_configs.append(
                        LocalWebDatasetConfig(
                            id=wds_config["id"],
                            path=wds_config["path"],
                            custom_metadata_fn=custom_metadata_fn
                        )
                    )

        return WebDatasetDataLoader(
            wds_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers,
            persistent_workers=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000)
        ).data_loader

    elif dataset_type == "latent_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            split_path = audio_dir_config.get("split_path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            
            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    split_path=split_path,
                )
            )

        train_set = LatentDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            input_type=dataset_config.get("input_type", "video"),
            fps=dataset_config.get("fps", 4),
            force_channels=force_channels
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)
    elif dataset_type == 'multimodal_dir':
        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            split_path = audio_dir_config.get("split_path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    split_path=split_path,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        train_set = MultiModalDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            input_type=dataset_config.get("input_type", "video"),
            fps=dataset_config.get("fps", 4),
            force_channels=force_channels
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=shuffle,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)