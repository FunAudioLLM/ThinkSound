import os
import requests
from tqdm import tqdm
import torch
import numpy as np
import laion_clap
from clap_module.factory import load_state_dict
import librosa
import pyloudnorm as pyln
from typing import List, Optional

# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def int32_to_int16_trunc(x):
    return x.astype(np.int16)

def round_half(x: float) -> float:
    return round(x * 2) / 2

class CLAPRewardModel(torch.nn.Module):
    def __init__(
        self,
        clap_model='630k-audioset-fusion-best.pt',
        device: Optional[str] = None,
    ):
        super(CLAPRewardModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        # load model
        if clap_model == 'music_speech_audioset_epoch_15_esc_89.98.pt':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
            clap_path = 'load/clap_score/music_speech_audioset_epoch_15_esc_89.98.pt'
            self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
        elif clap_model == 'music_audioset_epoch_15_esc_90.14.pt':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
            clap_path = 'load/clap_score/music_audioset_epoch_15_esc_90.14.pt'
            self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
        elif clap_model == 'music_speech_epoch_15_esc_89.25.pt':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt'
            clap_path = 'load/clap_score/music_speech_epoch_15_esc_89.25.pt'
            self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
        elif clap_model == '630k-audioset-fusion-best.pt':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
            clap_path = 'load/clap_score/630k-audioset-fusion-best.pt'
            self.model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
        else:
            raise ValueError('clap_model not implemented')

        # download clap_model if not already downloaded
        if not os.path.exists(clap_path):
            print('Downloading ', clap_model, '...')
            os.makedirs(os.path.dirname(clap_path), exist_ok=True)

            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(clap_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                    for data in response.iter_content(chunk_size=8192):
                        file.write(data)
                        progress_bar.update(len(data))

        # fixing CLAP-LION issue, see: https://github.com/LAION-AI/CLAP/issues/118
        pkg = load_state_dict(clap_path)
        pkg.pop('text_branch.embeddings.position_ids', None)
        self.model.model.load_state_dict(pkg)
        self.model.eval()



    def forward(self, audio, text):
        """
        Cosine similarity is computed between the LAION-CLAP text embedding of the given prompt and 
        the LAION-CLAP audio embedding of the generated audio. LION-CLAP: https://github.com/LAION-AI/CLAP
        
        This evaluation script assumes that audio_path files are identified with the ids in id2text.
        
        clap_score() evaluates all ids in id2text.

        GPU-based computation.

        Select one of the following models from https://github.com/LAION-AI/CLAP:
            - music_speech_audioset_epoch_15_esc_89.98.pt (used by musicgen)
            - music_audioset_epoch_15_esc_90.14.pt
            - music_speech_epoch_15_esc_89.25.pt
            - 630k-audioset-fusion-best.pt (our default, with "fusion" to handle longer inputs)

        Params:
        -- id2text: dictionary with the mapping between id (generated audio filenames in audio_path) 
                    and text (prompt used to generate audio). clap_score() evaluates all ids in id2text.
        -- audio_path: path where the generated audio files to evaluate are available.
        -- audio_files_extension: files extension (default .wav) in eval_path.
        -- clap_model: choose one of the above clap_models (default: '630k-audioset-fusion-best.pt').
        Returns:
        -- CLAP-LION score
        """

        
        #print(audio.shape)
        text = [a.item() for a in text]

        if text:   

            batch_size = 64
            text_emb = {}
            for i in range(0, audio.shape[0], batch_size):
                batch_ids = list(range(i, min(i+batch_size, audio.shape[0])))
                batch_texts = text[i:i+batch_size]
                with torch.no_grad():
                    embeddings = self.model.get_text_embedding(batch_texts, use_tensor=True)
                for id, emb in zip(batch_ids, embeddings):
                    text_emb[id] = emb

        else:
            raise ValueError('Must specify id2text')



        scores=[]
        for id in range(audio.shape[0]):

                with torch.no_grad():
                    Taudio= audio[id].cpu().numpy()
                    Taudio = np.mean(Taudio, axis=0)
                    Taudio = pyln.normalize.peak(Taudio, -1.0)
                    Taudio = Taudio.reshape(1, -1) # unsqueeze (1,T)
                    Taudio = torch.from_numpy(int16_to_float32((Taudio))).float()
                    audio_embeddings = self.model.get_audio_embedding_from_data(x = Taudio, use_tensor=True)
                cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_emb[id].unsqueeze(0), dim=1, eps=1e-8)[0]
                scores.append(cosine_sim.item())


        return scores