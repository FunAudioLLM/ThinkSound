import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from torchvision.utils import save_image

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VGGSound(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        *,
        tsv_path: Union[str, Path] = 'dataset/vggsound/split_txt/train_caption.csv',
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        save_dir: str = 'data/vggsound/video_latents_text/train'
    ):
        self.root = Path(root)
        
        # videos = sorted(os.listdir(self.root))
        # videos = set([Path(v).stem for v in videos])  # remove extensions
        videos = []
        self.labels = []
        self.cots = []
        self.videos = []
        missing_videos = []
        # read the tsv for subset information
        df_list = pd.read_csv(tsv_path, sep=',', dtype={'id': str}).to_dict('records')
        
        # 控制处理的行范围
        if start_row is not None and end_row is not None:
            df_list = df_list[start_row:end_row]
        
        for record in df_list:
            id = record['id']
            # if os.path.exists(f'{save_dir}/{id}.pth'): 
            #     continue
                # try:
                #     torch.load(f'{save_dir}/{id}.pth')
                #     continue
                # except:
                #     print(f'error load file: {save_dir}/{id}.pth')
                #     os.system(f'rm -f {save_dir}/{id}.pth')
            label = record['caption']
            # if id in videos:
            self.labels.append(label)
            self.cots.append(record['caption_cot'])
            # self.labels[id] = label
            self.videos.append(id)
            # else:
            #     missing_videos.append(id)

        log.info(f'{len(videos)} videos found in {root}')
        log.info(f'{len(self.videos)} videos found in {tsv_path}')
        log.info(f'{len(missing_videos)} videos missing in {root}')




    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        label = self.labels[idx]
        cot = self.cots[idx]
        data = {
            'id': video_id,
            'caption': label,
            'caption_cot': cot
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
#         root="data/vggsound/video/test",
#         tsv_path="data/vggsound/split_txt/temp.csv",
#         sample_rate=44100,
#         duration_sec=9.0,
#         audio_samples=397312,
#         start_row=0,
#         end_row=None,
#         save_dir="data/vggsound/video_latents_text/test"
#     )
# dataset[0]