import json
import logging
import re
from typing import List, Optional
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import sys

AXES_NAME = ["CE", "CU", "PC", "PQ"]
from audiobox_aesthetics.model.aes import AesMultiOutput


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from dataclasses import dataclass
@dataclass(eq=False)
class Normalize:
    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean



def int16_to_float32(x):
    return (x / 32767.0).to(torch.float32)

def make_inference_batch(
    input_wavs: list,
    hop_size=10,
    window_size=10,
    sample_rate=16000,
    pad_zero=True,
):
    wavs = []
    masks = []
    weights = []
    bids = []
    offset = hop_size * sample_rate
    winlen = window_size * sample_rate
    for bid, wav in enumerate(input_wavs):
        for ii in range(0, wav.shape[-1], offset):
            wav_ii = wav[..., ii : ii + winlen]
            wav_ii_len = wav_ii.shape[-1]
            if wav_ii_len < winlen and pad_zero:
                wav_ii = F.pad(wav_ii, (0, winlen - wav_ii_len))
            mask_ii = torch.zeros_like(wav_ii, dtype=torch.bool)
            mask_ii[:, 0:wav_ii_len] = True
            wavs.append(wav_ii)
            masks.append(mask_ii)
            weights.append(wav_ii_len / winlen)
            bids.append(bid)
    return wavs, masks, weights, bids




class AesPredictor(torch.nn.Module):
    def __init__(
        self,
        precision: str = "bf16",
        batch_size: int = 1,
        data_col: str = "path",
        sample_rate: int = 16000,
        eval_type: str = "All",
        device = "cuda"):

        super(AesPredictor, self).__init__()
        self.device=device
        self.type = eval_type
        self.sample_rate = sample_rate

        self.model = AesMultiOutput.from_pretrained("facebook/audiobox-aesthetics")


        self.model.to(self.device)
        self.model.eval()


        self.target_transform = {
            axis: Normalize(
                mean=self.model.target_transform[axis]["mean"],
                std=self.model.target_transform[axis]["std"],
            )
            for axis in AXES_NAME
        }

    def audio_resample_mono(self, data_list) -> List:
        wavs = []
        for ii, item in enumerate(data_list):
            sr =44100
            wav = item
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=sr,
                new_freq=self.sample_rate,
            )
            
            wavs.append(wav)
        return wavs

    def forward(self, batch):
        with torch.inference_mode():
            bsz = len(batch)
            wavs = self.audio_resample_mono(int16_to_float32(batch))
            wavs, masks, weights, bids = make_inference_batch(
                wavs,
                10,
                10,
                sample_rate=self.sample_rate,
            )

            # collate
            wavs = torch.stack(wavs).to(self.device)
            masks = torch.stack(masks).to(self.device)
            weights = torch.tensor(weights).to(self.device)
            bids = torch.tensor(bids).to(self.device)

            assert wavs.shape[0] == masks.shape[0] == weights.shape[0] == bids.shape[0]
            preds_all = self.model({"wav": wavs, "mask": masks})
            all_result = {}
            for axis in AXES_NAME:
                preds = self.target_transform[axis].inverse(preds_all[axis])
                weighted_preds = []
                for bii in range(bsz):
                    weights_bii = weights[bids == bii]
                    weighted_preds.append(
                        (
                            (preds[bids == bii] * weights_bii).sum() / weights_bii.sum()
                        ).item()
                    )
                all_result[axis] = weighted_preds
            # re-arrenge result
            all_rows = [
                dict(zip(all_result.keys(), vv)) for vv in zip(*all_result.values())
            ]
            #print(all_rows)
            if self.type == "PC":
                return [(rows["PC"])/20 for rows in all_rows]
            elif self.type == "CE":
                return [(rows["CE"])/20 for rows in all_rows]
            return [(rows["CE"]+rows["CU"]-rows["PC"]+rows["PQ"])/40 for rows in all_rows]
            

