from typing import Literal, Optional
import json
# import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
from ThinkSound.models.factory import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict
from ThinkSound.training.utils import copy_state_dict
from transformers import AutoModel
from transformers import AutoProcessor
from transformers import T5EncoderModel, AutoTokenizer
import logging


from data_utils.ext.synchformer import Synchformer

log = logging.getLogger()

def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_get_text_features(self, input_ids=None, attention_mask=None, position_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = text_outputs[0]
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features, last_hidden_state

    clip_model.get_text_features = new_get_text_features.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):
 
    def __init__(
        self,
        *, 
        vae_ckpt: Optional[str] = None,
        vae_config: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            self.clip_model = AutoModel.from_pretrained("metaclip-h14-fullcc2.5b")
            self.clip_model = patch_clip(self.clip_model)
            self.t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
            self.t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-xl")
            self.clip_processor = AutoProcessor.from_pretrained("metaclip-h14-fullcc2.5b")
            # self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
            #                                  std=[0.26862954, 0.26130258, 0.27577711])
            # self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'
        else:
            self.clip_model = None
            self.synchformer = None
            self.tokenizer = None

        if vae_ckpt is not None:
            with open(vae_config) as f:
                vae_config = json.load(f)
            self.vae = create_model_from_config(vae_config)
            print(f"Loading model checkpoint from {vae_ckpt}")
            # Load checkpoint
            copy_state_dict(self.vae, load_ckpt_state_dict(vae_ckpt,prefix='autoencoder.'))#,prefix='autoencoder.'
        else:
            self.tod = None

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)


    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        tokens = self.clip_processor(text=text, truncation=True, max_length=77, padding="max_length",return_tensors="pt").to(self.device)
        return self.clip_model.get_text_features(**tokens)

    @torch.inference_mode()
    def encode_t5_text(self, text: list[str]) -> torch.Tensor:
        assert self.t5_model is not None, 'T5 model is not loaded'
        assert self.t5_tokenizer is not None, 'T5 Tokenizer is not loaded'
        # x: (B, L)
        inputs = self.t5_tokenizer(text,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt").to(self.device)
        return self.t5_model(**inputs).last_hidden_state

    @torch.inference_mode()
    def encode_audio(self, x) -> torch.Tensor:
        x = self.vae.encode(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
