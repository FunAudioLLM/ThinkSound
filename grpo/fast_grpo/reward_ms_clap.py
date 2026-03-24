from msclap import CLAP as CLAPWrapper
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import requests
from tqdm import tqdm
import torch
import numpy as np
import librosa
import pyloudnorm as pyln
from typing import List, Optional

# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.float32) / 32767.0)

def float32_to_int16(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=-1.0, max=1.0)
    return (x * 32767.0).to(torch.int16)

def int32_to_int16_trunc(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.int16)

def round_half(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 2) / 2

class MSCLAPRewardModel(torch.nn.Module):
    def __init__(
        self,
        version = '2023',
        device: Optional[str] = None,
    ):
        super(MSCLAPRewardModel, self).__init__()

        self.device = device
        torch.cuda.set_device(self.device)
        # Load model (Choose between versions '2022' or '2023')
        # The model weight will be downloaded automatically if `model_fp` is not specified
        self.model = CLAPWrapper(version = version, use_cuda=True)
        #self.model.clap = self.model.clap.to(device)   

        




    def forward(self, audio, text):


        
        torch.cuda.set_device(self.device)
        text = [a.item() for a in text]
             
        audio = int16_to_float32(audio)

        scores = []
        for i in range(audio.shape[0]):
            text_embeddings = self.model.get_text_embeddings([text[i]])
            audio_curr = torch.mean(audio[i], dim=0, keepdim=True)
            preprocessed_audio = self.model.default_collate([audio_curr])
            audio_embeddings = self.model._get_audio_embeddings(preprocessed_audio)
            similarities = self.model.compute_similarity(audio_embeddings, text_embeddings)
            scores.append(similarities[0][0].item()/80)
            


        return scores
