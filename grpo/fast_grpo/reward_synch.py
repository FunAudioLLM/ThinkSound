import argparse
import subprocess
from pathlib import Path
from PIL import Image
import torch
import torchaudio
import torchvision
from omegaconf import OmegaConf
import os

import sys


from pathlib import Path
import torchvision
from concurrent.futures import ProcessPoolExecutor
import threading
from pathlib import Path
from typing import Tuple


Shape=None

def make_class_grid(leftmost_val, rightmost_val, grid_size, add_extreme_offset: bool = False,
                    seg_size_vframes: int = None, nseg: int = None, step_size_seg: float = None,
                    vfps: float = None):
    assert grid_size >= 3, f'grid_size: {grid_size} doesnot make sense. If =2 -> (-1,1); =1 -> (-1); =0 -> ()'
    grid = torch.from_numpy(np.linspace(leftmost_val, rightmost_val, grid_size)).float()
    if add_extreme_offset:
        assert all([seg_size_vframes, nseg, step_size_seg]), f'{seg_size_vframes} {nseg} {step_size_seg}'
        seg_size_sec = seg_size_vframes / vfps
        trim_size_in_seg = nseg - (1 - step_size_seg) * (nseg - 1)
        extreme_value = trim_size_in_seg * seg_size_sec
        grid = torch.cat([grid, torch.tensor([extreme_value])])  # adding extreme offset to the class grid
    return grid


def quantize_offset(grid: torch.Tensor, off_sec: float) -> Tuple[float, int]:
    '''Takes in the offset in seconds and snaps it onto the closest grid element.
    Returns the grid value and its index.'''
    closest_grid_el = (grid - off_sec).abs().argmin()
    return grid[closest_grid_el], closest_grid_el

def _read_and_reencode_one(path, vfps=25, in_size=256, get_meta=False, duration=9):
    path = str(path)


    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        '-i', path
    ]
    proc = subprocess.run(probe_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}")
    orig_w, orig_h = map(int, proc.stdout.strip().split(','))

    if orig_w <= orig_h:
        w = in_size
        h = int(orig_h * in_size / orig_w) // 2 * 2
    else:
        h = in_size
        w = int(orig_w * in_size / orig_h) // 2 * 2

    frame_size = w * h * 3

    cmd = [
        'ffmpeg',
        '-i', path,
        '-t', str(duration),
        '-vf', f"fps={vfps},scale={w}:{h},crop='trunc(iw/2)*2':'trunc(ih/2)*2'",
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-'
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    

    frames = []
    stderr_data = b""

    while True:
        raw = pipe.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        
        frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        frames.append(frame)

    pipe.stdout.close()

    # Read stderr separately, then wait for process
    stderr_data = pipe.stderr.read()
    pipe.wait()  # Use wait() instead of communicate()

    if pipe.returncode != 0:
        print(stderr_data.decode())
        raise RuntimeError(f"ffmpeg failed on {path}")

    if not frames:
        print(stderr_data.decode())
        raise RuntimeError(f"No frames extracted, check ffmpeg output for {path}")

    video_tensor = (
        torch.tensor(np.stack(frames), dtype=torch.uint8)
        .permute(0, 3, 1, 2)
    )

    if get_meta:
        meta = {
            'video': {'fps': [vfps]},
            'audio': {'framerate': [16000]},
        }
        return video_tensor, meta
    else:
        return video_tensor



def get_videos_with_meta(paths, vfps=25, in_size=256, workers=4, get_meta=False):
    single_input = False
    if isinstance(paths, (str, Path)):
        paths = [paths]
        single_input = True

    func = partial(_read_and_reencode_one, vfps=vfps, in_size=in_size, get_meta=get_meta)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(func, paths))

    if get_meta:
        videos = [r[0] for r in results]
        metas = [r[1] for r in results]
        return (videos[0], metas[0]) if single_input else (videos, metas)
    else:
        return results[0] if single_input else results



def _delete_file(path):
    try:
        os.remove(path)
    except Exception as e:
        print(f"Failed to delete {path}: {e}")

def async_delete(paths):

    for p in paths:
        t = threading.Thread(target=_delete_file, args=(p,), daemon=True)
        t.start()

def int16_to_float32(x):
    return (x / 32767.0).to(torch.float32)

import subprocess
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def _reencode_one(path, vfps=25, in_size=256,duration=9):
    path = Path(path)
    safe_stem = path.stem.replace(" ", "_")
    new_path = Path.cwd() / 'vis' / f"{safe_stem}_{vfps}fps_{in_size}side.mp4"
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    if os.path.exists(new_path):
        return new_path
    
    duration = 9 if len(safe_stem)==18 else None
    
    cmd = 'ffmpeg'
    cmd += ' -hide_banner -loglevel panic'
    if duration is not None:
        cmd += f' -t {duration}'
    cmd += f' -y -i {path}'
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += ' -an'
    cmd += f' {new_path}'
    subprocess.call(cmd.split())

    return new_path

def reencode_helper(p, vfps, in_size):
    return _reencode_one(p, vfps, in_size)

def reencode_videos(paths, vfps=25, in_size=256, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        func = partial(reencode_helper, vfps=vfps, in_size=in_size)
        results = list(ex.map(func, paths))
    return results

import torchvision
from pathlib import Path
import time

def _read_one(path, start_sec=0, end_sec=None, get_meta=False, max_retries=3, retry_delay=0.5):
    global Shape
    path = Path(path)
    last_err = None
    rgb = None
    meta = None

    for attempt in range(max_retries):
        try:
            rgb, _, meta = torchvision.io.read_video(
                str(path),
                start_sec,
                end_sec,
                'sec',
                output_format='TCHW'
            )

            if rgb is not None and rgb.numel() > 0 and rgb.shape[0] > 0:
                break
        except Exception as e:
            last_err = e

        time.sleep(retry_delay)
        rgb, meta = None, None

    meta = {
        'video': {'fps': [25]},
        'audio': {'framerate': [16000]}
    }

    if rgb is None or rgb.numel() == 0:
        rgb = torch.randint(0, 256, Shape, dtype=torch.uint8)
        return  (rgb, meta)
    #if torch.all(rgb == 0):
    #    print("all blank")
    #    rgb = torch.randint(0, 256, rgb.shape, dtype=torch.uint8)

    Shape = rgb.shape
    return (rgb, meta) if get_meta else rgb


def get_video(paths, get_meta=False, start_sec=0, end_sec=None, workers=12):
    single_input = False
    if isinstance(paths, (str, Path)):
        paths = [paths]
        single_input = True

    func = partial(_read_one, start_sec=start_sec, end_sec=end_sec, get_meta=get_meta)

    #with ProcessPoolExecutor(max_workers=workers) as ex:
    #    results = list(ex.map(func, paths))
    results = [func(path) for path in paths]
    
    if get_meta:
        rgbs = [r[0] for r in results]
        metas = [r[1] for r in results]
        return (rgbs, metas) if not single_input else (rgbs[0], metas[0])
    else:
        return results if not single_input else results[0]




def decode_single_video_prediction(off_logits, grid, item):


    off_probs = torch.softmax(off_logits, dim=-1)
    k = min(off_probs.shape[-1], 5)
    topk_logits, topk_preds = torch.topk(off_logits, k)
    # remove batch dimension
    assert len(topk_logits) == 1, 'batch is larger than 1'
    topk_logits = topk_logits[0]
    topk_preds = topk_preds[0]
    off_logits = off_logits[0]
    off_probs = off_probs[0]
    for target_hat in topk_preds:
        print(f'p={off_probs[target_hat]:.4f} ({off_logits[target_hat]:.4f}), "{grid[target_hat]:.2f}" ({target_hat})')
    return off_probs

def decode_max_grid(off_logits_list, grid):
    max_grids = []
    for off_logits in off_logits_list:
        target_hat = torch.argmax(off_logits, dim=-1).item()
        max_grids.append(2-abs(grid[target_hat]))

    return max_grids

def patch_config(cfg):
    # the FE ckpts are already in the model ckpt
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    # old checkpoints have different names
    cfg.model.params.transformer.target = cfg.model.params.transformer.target\
                                             .replace('.modules.feature_selector.', '.sync_model.')
    return cfg

class synchpredict(torch.nn.Module):
    def __init__(
        self,
        exp_name= "24-01-04T16-39-21",
        device = "cuda"):
        import absl.logging
        absl.logging.set_verbosity('error')
        initial_modules = set(sys.modules.keys())


        sys.path.insert(0, 'Synchformer')
        sys.path.insert(0, 'Synchformer/model/modules/feat_extractors/visual')
        try:
            from utils.utils import check_if_file_exists_else_download
            from scripts.train_utils import get_model, get_transforms, prepare_inputs
            

            super(synchpredict, self).__init__()
            self.vfps = 25
            self.afps = 16000
            self.in_size = 256
            self.exp_name= exp_name
            self.cfg_path = f'ckpts/{exp_name}/cfg-{exp_name}.yaml'
            self.ckpt_path = f'ckpts/{exp_name}/{exp_name}.pt'

            # if the model does not exist try to download it from the server
            
            check_if_file_exists_else_download(self.cfg_path)
            check_if_file_exists_else_download(self.ckpt_path)

            # load config
            self.cfg = OmegaConf.load(self.cfg_path)

            # patch config
            self.cfg = patch_config(self.cfg)

            self.device = device
            # load the model
            _, self.model = get_model(self.cfg, device)
            ckpt = torch.load(self.ckpt_path, map_location=torch.device('cpu'),weights_only=False)
            self.model.load_state_dict(ckpt['model'])
            self.model.eval()
            self.get_transforms = get_transforms
            self.prepare_inputs = prepare_inputs

        finally:
            sys.path.pop(0)
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

    def forward(self, batch, metadata):
        paths = []
        for meta in metadata:
            video_path = meta.get('video_path', None)
            if video_path and os.path.exists(video_path.item()):
                paths.append(video_path.item())
            else:
                raise FileNotFoundError(f"Video file for id {meta['id']} not found.")

        # vid_paths = reencode_videos(paths, self.vfps, self.in_size, workers=12)
        # rgb, metas = get_video(vid_paths, get_meta=True)
        
        rgb_list = []
        metas = []
        for p in paths:
            video_tensor, meta = _read_and_reencode_one(
                p, vfps=self.vfps, in_size=self.in_size, get_meta=True
            )
            rgb_list.append(video_tensor)
            metas.append(meta)
        # ======================

        B, C, L = batch.shape
        assert C == 2, "Input must be stereo [B,2,L]"

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resampler = torchaudio.transforms.Resample(
                orig_freq=44100, new_freq=self.afps,
                resampling_method='sinc_interpolation',
                lowpass_filter_width=64
            ).to(self.device)


        audio_resampled = resampler(int16_to_float32(batch))
        audios = audio_resampled.mean(dim=1)

        item = [dict(
            video=rgb_list[i], audio=audios[i].to("cpu"), meta=metas[i], path=paths[i], split='test',
            targets={'v_start_i_sec': 0.0, 'offset_sec': 0.0},
        ) for i in range(len(batch))]

        max_off_sec = self.cfg.data.max_off_sec
        num_cls = self.cfg.model.params.transformer.params.off_head_cfg.params.out_features
        grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)

        test_transform = self.get_transforms(self.cfg, ['test'])['test']

        processed = []
        failed_mask = []
        for it in item:
            try:
                processed.append(test_transform(it))
                failed_mask.append(False)
            except Exception as e:
                print(f"[WARN] transform failed: {it}, {e}, in cuda {self.device}")
                processed.append(None)
                failed_mask.append(True)

        valid_items = [p for p in processed if p is not None]
        if len(valid_items) == 0:
            return [0 for _ in item]

        batch = torch.utils.data.default_collate(valid_items)
        aud, vid, targets = self.prepare_inputs(batch, self.device)

        with torch.set_grad_enabled(False):
            with torch.autocast('cuda', enabled=self.cfg.training.use_half_precision):
                _, logits = self.model(vid, aud)

        scores = decode_max_grid(logits, grid)
        scores = [s/4 for s in scores]

        final_scores = []
        idx = 0
        for fail in failed_mask:
            if fail:
                final_scores.append(0)
            else:
                final_scores.append(scores[idx])
                idx += 1

        return final_scores




