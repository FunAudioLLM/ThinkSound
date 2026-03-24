import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import numpy as np

from tqdm import tqdm



import torch

import torch.nn.functional as F


import tempfile
import torch
import torchaudio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Optional
import logging
from datetime import datetime
import traceback
import torch.nn as nn


def int16_to_float32(x):
    return (x / 32767.0).to(torch.float32)


import uuid


class FastAudioProcessor:
    def __init__(self, device ,num_threads=2, tmp_dir="tmp"):
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self._tmp_files = []
        self.tmp_dir = tmp_dir
        self.device = device
        os.makedirs(self.tmp_dir, exist_ok=True)

    def process_audio(self, audio: torch.Tensor, sr: int) -> str:
        audio = audio.unsqueeze(0) if audio.ndim == 1 else audio
        unique_id = f"{int(time.time()*1e6)}_{uuid.uuid4().hex}"
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix=f"rank{self.device.index}_{unique_id}_",
            dir=self.tmp_dir,
            delete=False
        ) as f:
            path = f.name

        torchaudio.save(path, audio.cpu(), sr)
        self._tmp_files.append(path)

        with open(path, "rb") as f:
            os.fsync(f.fileno())
        return path

    def cleanup(self):
        remaining_files = []
        for path in self._tmp_files:
            if os.path.basename(path).startswith(f"rank{self.device}_"):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            else:
                remaining_files.append(path)
        self._tmp_files = remaining_files

class SuppressOutput:
    def __enter__(self):
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.old_stdout = os.dup(self.stdout_fd)
        self.old_stderr = os.dup(self.stderr_fd)
        
        self.devnull = open(os.devnull, 'w')
        
        os.dup2(self.devnull.fileno(), self.stdout_fd)
        os.dup2(self.devnull.fileno(), self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.old_stdout, self.stdout_fd)
        os.dup2(self.old_stderr, self.stderr_fd)
        os.close(self.old_stdout)
        os.close(self.old_stderr)
        self.devnull.close()

def calculate_mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

def calculate_mae(arr1, arr2):
    return np.mean(np.abs(arr1 - arr2))

def softmax_normalize(x):
    return F.softmax(torch.tensor(x), dim=-1)

def normalize(distribution):
    """
    Normalize the input list to a probability distribution.
    :param distribution: Array or list
    :return: Normalized probability distribution
    """
    distribution = np.array(distribution)
    total = np.sum(distribution)
    if total == 0:
        return np.ones_like(distribution) / len(distribution)
    return distribution / total

def kl_divergence(p, q):
    """
    Calculate the KL divergence between two probability distributions.
    :param p: Probability distribution P, in array form
    :param q: Probability distribution Q, in array form
    :return: KL divergence value
    """
    p = normalize(p)
    q = normalize(q)
    epsilon = 1e-10
    q = np.clip(q, epsilon, 1.0)
    p = np.clip(p, epsilon, 1.0)

    return np.sum(p * np.log(p / q))


from argparse import Namespace

def get_config():
    return Namespace(
        # basic configuration
        exp="test101",
        epochs=100,
        start_epoch=0,
        resume='ckpts/FreeMusic-StereoCRW-1024.pth.tar',
        resume_optim=False,
        save_step=1,
        valid_step=1,

        # Dataloader parameter
        max_sample=-1,
        repeat=1,
        num_workers=0,
        batch_size=2,

        # network parameters
        setting='stereocrw_binaural',
        backbone='resnet9',

        # optimizer parameters
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        optim="Adam",
        schedule="cos",

        # Loss parameters
        loss_type="MSE",
        no_bn=False,
        aug_wave=False,
        regular_scaling=False,
        shift_wave=False,
        larger_shift=False,
        aug_img=False,
        normalized_rms=False,
        valid_by_step=False,
        test_mode=False,
        no_resample=False,
        add_color_jitter=False,

        patch_size=None,
        patch_stride=1,
        patch_num=512,
        skip_node=False,
        fake_right=False,

        wav2spec=True,
        tau=0.05,

        cycle_num=1,
        crw_rate=1.0,
        synthetic_rate=0.0,
        teacher_rate=0.0,
        clip_length=6,

        large_feature_map=False,
        add_sounds=0,
        max_weight=0.0,

        smooth=0.0,
        noiseSNR=None,
        bidirectional=False,

        no_baseline=True,
        baseline_type="mean",

        mode="mean",
        select="soft_weight",
        gcc_fft=3840,

        cycle_filter=False,
        img_feat_scaling=None,
        max_delay=None,
        correspondence_type="mean",

        add_reverb=False,
        add_mixture=False,
        ignore_speaker=False,
        crop_face=False,
        pretrained="",

        same_vote=False,
        same_noise=False,
        finer_hop=False,
    )



class itdpredict(torch.nn.Module):
    def __init__(
        self,
        mode = "mse",
        num_threads = 4,
        device = "cuda"):
        super(itdpredict, self).__init__()

        initial_modules = set(sys.modules.keys())

        EVAL_IDT_PATH = 'stereocrw'
        sys.path.insert(0, EVAL_IDT_PATH)

        try:
            import data.stereo_audio_loader as _sal_module
            _original_get_list_sample = _sal_module.StereoAudioDataset.get_list_sample
            def _patched_get_list_sample(self, list_sample):
                if isinstance(list_sample, list):
                    return list_sample
                return _original_get_list_sample(self, list_sample)
            _sal_module.StereoAudioDataset.get_list_sample = _patched_get_list_sample

            import data.binaural_loader as _binaural_loader
            def _patched__getitem__(self, index):
                info = self.list_sample[index]
                audio_path = info
                meta = torchaudio.info(audio_path)
                audio_sample_rate = meta.sample_rate
                ratio = 1.2 if self.pr.clip_length >= 0.48 else 2
                audio_start = 0
                audio_end = int(audio_start + ratio * self.pr.clip_length * audio_sample_rate)
                audio, audio_rate = self.read_audio(audio_path, start=audio_start, stop=audio_end)
                audio = torch.from_numpy(audio.copy()).float()
                if not self.add_noise_with_snr == None:
                    audio = self.addGaussianSNR(audio, self.add_noise_with_snr, index)
                if_fakeright = False
                patch_size = int(self.pr.clip_length * audio_rate)
                lefts, rights, audio, shift_offset = self.generate_audio(audio, audio_rate, if_fakeright, index, patch_size, add_noise=False, audio_start=0)
                delay_time = self.create_delay_matrix(self.pr.patch_num, audio_rate)
                batch = {
                    'audio': audio,
                    'left_audios': lefts,
                    'right_audios': rights,
                    'audio_rate': torch.tensor(audio_rate),
                    'delay_time': delay_time,
                }
                if self.setting.find('pgccphat') != -1:
                    pgcc_phat = self.calc_pgcc_phat(audio, audio_rate)
                    batch['pgcc_phat'] = pgcc_phat
                return batch
            def new__init__(self, args, pr, list_sample, split='train'):
                super(_binaural_loader.BinauralAudioDataset, self).__init__(args, pr, list_sample, split)

            _binaural_loader.BinauralAudioDataset.__getitem__ = _patched__getitem__
            _binaural_loader.BinauralAudioDataset.__init__ = new__init__

            import models.audio_encoder as _audio_encoder
            def new_wav2stft(self, audio):
                audio = audio.squeeze(-2)
                audio_size = audio.size()
                spec = self.trans(audio)
                if spec.is_complex():
                    spec = torch.view_as_real(spec)
                spec = spec.permute(0, 1, 4, 3, 2)[..., :-1, :-1]
                return spec
            _audio_encoder.AudioEncoder.wav2stft = new_wav2stft

            from utils import torch_utils
            self.torch_utils = torch_utils

            from config import params
            self.params = params

            from vis_scripts.vis_functs import predict, gcc_phat, update_param
            from vis_scripts.vis_functs import inference_crw_itd_with_cycle, inference_crw_itd_simple
            
            def estimate_crw_itd(args, pr, aff_L2R, delay_time, no_postprocess=False):
                if args.cycle_filter: 
                    crw_itds = inference_crw_itd_with_cycle(args, pr, aff_L2R, delay_time)
                else:
                    crw_itds = inference_crw_itd_simple(args, pr, aff_L2R, delay_time)
                if no_postprocess: 
                    return crw_itds
                itds = []
                for i in range(aff_L2R.shape[0]):
                    curr_itds = crw_itds[i]
                    if args.mode == 'mean':
                        itd = torch.mean(curr_itds)
                    elif args.mode == 'ransac':
                        itd = ransac_like_pick(args, pr, curr_itds) 
                    itds.append(itd)
                itds = torch.stack(itds)
                pool = nn.AdaptiveAvgPool1d(256)
                crw_itds = pool(crw_itds.unsqueeze(1)).squeeze(1)
                return itds,crw_itds

            self.estimate_crw_itd = estimate_crw_itd
            
            import importlib
            models_module = importlib.import_module('models')
            self.models = models_module
            
            self.predict = predict
            self.gcc_phat = gcc_phat
            self.update_param = update_param

        finally:
            sys.path.pop(0)
            current_modules = set(sys.modules.keys())
            newly_imported = current_modules - initial_modules
            project_keywords = ['utils', 'scripts', 'models', 'config', 'data', 'vis_scripts']
            for mod in newly_imported:
                is_project_module = any(k in mod for k in project_keywords)
                mod_obj = sys.modules.get(mod)
                mod_file = getattr(mod_obj, '__file__', '') or ''
                if is_project_module and 'site-packages' not in mod_file:
                    sys.modules.pop(mod, None)
            
            # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Cleaned {len(newly_imported)} modules from stereocrw.")

        self.mode = mode
        self.device = device

        
        #parser = init_args(return_parser=True)
        self.args = get_config()
        fn = getattr(params, self.args.setting)
        self.pr = fn()
        #self.pr.dataloader = 'SingleVideoDataset'
        self.update_param(self.args, self.pr)

        self.audio_processor = FastAudioProcessor(device ,num_threads = num_threads)
        #monkey patching for version compatibility
        import torchvision.models.resnet as _resnet_module
        _original_resnet = _resnet_module._resnet
        def _patched_resnet(arch, block, layers, pretrained=False, progress=True, **kwargs):
            if not pretrained:
                return _original_resnet(block=block, layers=layers, weights=None, progress=progress, **kwargs)
            else:
                return _original_resnet(arch, block, layers, pretrained=False, progress=True, **kwargs)
        _resnet_module._resnet = _patched_resnet
        self.net = self.models.__dict__[self.pr.net](self.args, self.pr, device=device).to(device)
        _resnet_module._resnet = _original_resnet

        self.criterion = self.models.__dict__[self.pr.loss](self.args, self.pr, device)


    def _inference(self, args, pr, net, criterion, data_set, data_loader, device='cuda'):
        crw_itds = []
        feat_itds = []
        baseline_itds = []
        args.no_baseline = False
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                audio = batch['audio']
                audio_rate = batch['audio_rate']
                delay_time = batch['delay_time'].to(device)
                out = self.predict(args, pr, net, batch, device)
                aff_L2R = criterion.inference(out, softmax=False)
                crw_itd, feat_itd = self.estimate_crw_itd(args, pr, aff_L2R, delay_time)
                crw_itds.append(crw_itd)
                feat_itds.append(feat_itd)
                for i in range(aff_L2R.shape[0]):
                    curr_audio = audio[i]
                    if args.no_baseline:
                        baseline_itd = 0
                    else:
                        baseline_itd = self.gcc_phat(args, pr, curr_audio, fs=audio_rate[i].item(), max_tau=pr.max_delay, interp=1)
                    baseline_itds.append(baseline_itd)
        crw_itds = torch.cat(crw_itds, dim=-1).data.cpu().numpy() * 1000
        baseline_itds = np.array(baseline_itds) * 1000
        feat_itds = torch.cat(feat_itds, dim=0)
        return crw_itds, baseline_itds, feat_itds

    def get_itds(self, files):
        self.pr.list_test = files
        with SuppressOutput():
            test_dataset, test_loader = self.torch_utils.get_dataloader(self.args, self.pr, split='test', shuffle=False, drop_last=False)
        crw_itds, baseline_itds, feat_itds = self._inference(self.args, self.pr, self.net, self.criterion, test_dataset, test_loader, self.device)
        return crw_itds, baseline_itds, feat_itds

    def forward(self, batch, metadata):

        #self.audio_processor.cleanup()
        futures = []
        batchsize = batch.shape[0]
        tmp_files = []
        for i in range(batchsize):
            future = self.audio_processor.executor.submit(
                self.audio_processor.process_audio,
                batch[i],
                44100
            )
            futures.append(future)
        file_paths = [future.result() for future in futures]

        tmp_files.extend(file_paths)

        gt_filepath = []
        for meta in metadata:
            audio_path = meta.get('audio_path', None)
            if audio_path and os.path.exists(audio_path.item()):
                gt_filepath.append(audio_path.item())
            else:
                raise FileNotFoundError(f"Audio file not found for id={meta.get('id')}, path={audio_path}")


        infered_crw_itds, infered_baseline_itds, infered_feat_itds = self.get_itds(file_paths)
        gt_crw_itds, gt_infered_baseline_itds, gt_feat_itds = self.get_itds(gt_filepath)


        score = []

        if self.mode == "mse":
            for i in range(batchsize):
            
                feat1_crw = infered_crw_itds[i].mean()
                feat2_crw = gt_crw_itds[i].mean()
                
                # print(feat1_crw, feat2_crw)
                mse_crw = 5*calculate_mse(feat1_crw, feat2_crw)
                #print(mse_crw)

                # Calculate for GCC mode
                feat1_gcc = infered_baseline_itds[i].mean()
                feat2_gcc = gt_infered_baseline_itds[i].mean()
                mse_gcc = 5*calculate_mse(feat1_gcc, feat2_gcc)
                #print(mse_gcc,mse_crw)
                score.append(-(mse_crw/2+mse_gcc)/2)

        #self.audio_processor.cleanup()
        
        for f in tmp_files:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        return score