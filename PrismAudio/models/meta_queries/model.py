# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
import math
from typing import List
import typing as tp
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true" 
USE_AUDIO_IN_VIDEO_RATIO = 1.0

from torch import nn
from torchvision import transforms as v2

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor, Qwen2Config

import time
from diffusers.models.normalization import RMSNorm

from transformers.video_utils import load_video

from .transformer_encoder import Qwen2Encoder

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
def to_device(
    data: Any, 
    device: Union[str, torch.device, int], 
    dtype: Optional[torch.dtype] = None,  # 新增
    non_blocking: bool = False
) -> Any:
    """Move inputs to a device and optionally convert dtype"""
    if isinstance(data, Mapping):
        return type(data)({
            k: to_device(v, device, dtype, non_blocking) 
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        return type(data)(
            to_device(v, device, dtype, non_blocking) 
            for v in data
        )
    elif isinstance(data, torch.Tensor):
        tensor = data.to(device=device, non_blocking=non_blocking)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor
    else:
        return data
    
VIDEO_MIN_PIXELS=224*224
VIDEO_MAX_PIXELS=224*224

import torch.distributed as dist
def print_memory_summary(prefix: str = ""):

    if not torch.cuda.is_available():
        return
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    usage = (allocated / total * 100) if total > 0 else 0
    
    print(f"[Rank {rank}] {prefix} | GPU Memory: {allocated:.2f}/{total:.2f} GB ({usage:.1f}%)")


class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        diffusion_model_id: str = None,
        num_metaqueries: int = None,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 2560,
        connector_num_hidden_layers: int = None,
        system_prompt: str = "You will be given an video and its caption. Please describe the content of the video in detail in your own words.",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.diffusion_model_id = diffusion_model_id
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.system_prompt = system_prompt

import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F



default_config = MLLMInContextConfig()

class MLLMInContext(PreTrainedModel):

    def __init__(
        self,
        output_dim: int,
        query_len: int,
        llm_id = "qwen_omni",
        connection_layers=12,
        config: MLLMInContextConfig = default_config,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config
        config.num_metaqueries = query_len
        config.connector_num_hidden_layers = connection_layers
        print("use meta queries: ",query_len,flush=True)

        if llm_id == "qwen_vl":
            config.mllm_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif llm_id == "qwen_omni":
            config.mllm_id = "Qwen/Qwen2.5-Omni-3B"
        else:
            raise ValueError(f"Unsupported model: {llm_id}")

        if "Qwen2.5-VL" in config.mllm_id:
            from .models.qwen25VL import (
                Qwen2_5_VLForConditionalGeneration
            )
            self.mllm_type = "qwenvl"
        elif "Qwen2.5-Omni" in config.mllm_id:
            from .models.qwen25omni import (
                Qwen2_5OmniForConditionalGeneration
            )
            self.mllm_type = "qwenomni"
        elif "Qwen" in config.mllm_id:
            self.mllm_type = "qwenlm"
        elif "Llama" in config.mllm_id:
            self.mllm_type = "llamaml"
        else:
            self.mllm_type = "llavaov"

        if self.mllm_type == "qwenvl":
            self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16
            )
            self.mllm_backbone.model.config.use_sliding_window = False
            self.mllm_backbone.model.config.sliding_window = None
            #print(self.mllm_backbone.model)


            self._freeze_mllm_backbone()
            
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook)
            self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
            self.mllm_backbone.lm_head = nn.Identity()

            self.tokenizer = AutoProcessor.from_pretrained(
                config.mllm_id, video_min_pixels=VIDEO_MIN_PIXELS, video_max_pixels=VIDEO_MAX_PIXELS,use_fast=True,min_pixels=224*224,max_pixels=288*288
            )
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
            #self.tokenizer.image_processor.size = {
            #    "height": 224,
            #    "width": 224
            #}
            # 3B 2048
            # 7B 3584
            self.tokenizer.system_prompt = config.system_prompt
        elif self.mllm_type == "qwenomni":
            self.mllm_backbone = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16
            )
            #self.mllm_backbone.disable_talker()
            self.mllm_backbone.thinker.model.config.use_sliding_window = False
            self.mllm_backbone.thinker.model.config.sliding_window = None
            self._freeze_mllm_backbone()
            
            num_embeddings = self.mllm_backbone.thinker.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.thinker.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.thinker.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.thinker.model.embed_tokens.weight.register_hook(freeze_hook)
            self.mllm_hidden_size = self.mllm_backbone.thinker.model.config.hidden_size
            self.mllm_backbone.thinker.lm_head = nn.Identity()

            self.tokenizer = AutoProcessor.from_pretrained(
                config.mllm_id, video_min_pixels=VIDEO_MIN_PIXELS, video_max_pixels=VIDEO_MAX_PIXELS,use_fast=True,min_pixels=224*224,max_pixels=288*288
            )
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
            self.tokenizer.system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")

        

        self.tokenizer.mllm_type = self.mllm_type
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        
        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id
        if config.num_metaqueries > 0:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<pad_token_{i}>"
                        for i in range(num_embeddings - len(tokenizer))
                    ]
                }
            )
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_audio>", "<end_of_audio>"]
                    + [f"<audio{i}>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_audio>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_audio>")

        #self.mllm_backbone = torch.compile(self.mllm_backbone)

        self.connector_in_dim = self.mllm_hidden_size
        self.connector_out_dim = output_dim

        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(1.0)

        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.connector_in_dim,
                intermediate_size=self.connector_in_dim * 4,
                num_hidden_layers=config.connector_num_hidden_layers,
                num_attention_heads=self.connector_in_dim // 64,
                num_key_value_heads=self.connector_in_dim // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
            except:
                pass
            if not isinstance(self.connector, nn.Identity):
                for module in self.connector:
                    if isinstance(module, Qwen2Encoder):
                        module.gradient_checkpointing_enable({"use_reentrant": False})

    def _freeze_mllm_backbone(self):

        print("\nFreeze MLLM backbone...")

        for param in self.mllm_backbone.parameters():
            param.requires_grad = False

        if self.config.num_metaqueries > 0:
            if hasattr(self.mllm_backbone,"model"):
                embed_tokens = self.mllm_backbone.model.embed_tokens
                embed_tokens.weight.requires_grad = True
            elif hasattr(self.mllm_backbone,"thinker"):
                embed_tokens = self.mllm_backbone.thinker.model.embed_tokens
                embed_tokens.weight.requires_grad = True


            

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    def get_resize_fn(self):
        return self.resize_fn

    @staticmethod
    @torch.no_grad()
    def tokenize(
        tokenizer, caption, video = None,audio = None, text_response=None, add_generation_prompt=True
    ):
        #print(video)
        if not isinstance(caption, List):
            caption = [caption]

        if video is not None and not isinstance(video, List):
            video = [video]
        if audio is not None and not isinstance(audio, List):
            audio = [audio]

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                        tokenizer.system_prompt
                        if tokenizer.mllm_type == "qwenlm"
                        else [{"type": "text", "text": tokenizer.system_prompt}]
                    ),
                },
            ]
            if tokenizer.system_prompt is not None
            else []
        )

        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
            suffix = ""
        elif tokenizer.mllm_type=="qwenvl":
            suffix = (
                "\n<begin_of_audio>"
                + "".join([f"<audio{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_audio><|im_end|>"
            )
        elif tokenizer.mllm_type=="qwenomni":
            suffix = (
                "\n<begin_of_audio>"
                + "".join([f"<audio{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_audio><|im_end|>"
            )

        caption = [
            tokenizer.decode(
                tokenizer(
                    text=cap, 
                    return_tensors="pt", 
                    padding="max_length",
                    max_length=tokenizer.max_input_text_tokens,
                    truncation=True
                ).input_ids[0]
            )
            for cap in caption
        ]

        if audio is not None:
            #print("audio",audio[0].shape,audio)
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, aud in enumerate(audio):
                if aud is not None and not isinstance(aud, list):
                    audio[i] = [aud]

        if video is not None:
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, vid in enumerate(video):
                if vid is not None and not isinstance(vid, list):
                    #print("vid shape",vid.shape,flush=True)
                    video[i] = [vid]

            # Resize each image in each batch if resize_fn is not None
            if tokenizer.resize_fn is not None:
                video = [
                    [tokenizer.resize_fn(sub_img) for sub_img in imgs] if imgs else None
                    for imgs in video
                ]
            if tokenizer.mllm_type == "qwenvl":
                conversations = [
                    prefix
                    + [
                        {
                            "role": "user",
                            "content": (
                                [{"type": "video"} for _ in vids]
                                + [{"type": "text", "text": cap}]
                                if vids
                                else [{"type": "text", "text": cap}]
                            ),
                        },
                    ]
                    for cap, vids in zip(caption, video)
                ]
                kwargs = {"videos": [imgs for imgs in video if imgs]}
            if tokenizer.mllm_type == "qwenomni":
                conversations = [
                    prefix
                    + [
                        {
                            "role": "user",
                            "content": (
                                [{"type": "video"} for vid in vids] if vids else []
                                + [{"type": "text", "text": cap}]
                            ),
                        },
                    ]
                    for cap, vids, auds in zip(caption, video, audio)
                ]
                kwargs = {"videos": [vid for vids in video for vid in vids],
                          "audio": [aud for auds in audio for aud in auds]}
                #print("conversations",conversations)
        elif tokenizer.mllm_type in ["qwenlm", "llamaml"]:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": cap,
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        else:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cap}],
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()


        prompts = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]
        if tokenizer.mllm_type=="qwenomni":
            prompts = [item for prompt in prompts for item in prompt]
        #print(prompts,flush=True)

        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]
        if tokenizer.num_metaqueries > 0:
            prompts = [p + suffix for p in prompts]

        #print("prompts",prompts)
        #print("kwargs",kwargs)
        use_audio_in_video = random.random() < USE_AUDIO_IN_VIDEO_RATIO
        #use_audio_in_video = True
        text_inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            videos_kwargs={"fps": 1, "use_audio_in_video": use_audio_in_video},
            **kwargs,
        )
        #print("text_inputs",text_inputs,flush=True)
        #print("input_ids",text_inputs["input_ids"].tolist(),flush=True)

        return text_inputs


    def encode_condition(
        self, input_ids, attention_mask, **kwargs
    ):
        if self.mllm_type == "llavaov":
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                **kwargs,
                attention_mask=attention_mask,
            ).logits
        elif self.mllm_type in ["qwenvl"]:

            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                **kwargs,
                attention_mask=attention_mask,
            ).logits
         
        elif self.mllm_type in ["qwenomni"]:
            prompt_embeds = self.mllm_backbone.thinker(
                input_ids=input_ids,
                **kwargs,
                attention_mask=attention_mask,
            ).logits
        elif self.mllm_type in ["qwenlm", "llamaml"]:
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")

        if self.tokenizer.num_metaqueries > 0:
            # Get positions for all sequences in batch at once
            boi_pos = torch.where(input_ids == self.boi_token_id)[1]
            eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]

            # Create mask for selecting tokens between BOI and EOI
            batch_size, seq_len = input_ids.shape
            indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
                batch_size, -1
            )

            

            if boi_pos.shape[0] == batch_size and eoi_pos.shape[0] == batch_size:
                mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])
                prompt_embeds = prompt_embeds[mask].view(
                    batch_size, -1, prompt_embeds.size(-1)
                )
                attention_mask = attention_mask[mask].view(batch_size, -1)
            else: 
                print(f"[DEBUG] boi_pos.shape[0]={boi_pos.shape[0]}, eoi_pos.shape[0]={eoi_pos.shape[0]}")
                print(f"[DEBUG] boi_pos={boi_pos}")
                print(f"[DEBUG] eoi_pos={eoi_pos}",flush=True)
                prompt_embeds = torch.zeros(
                    batch_size, 
                    self.tokenizer.num_metaqueries, 
                    prompt_embeds.size(-1),
                    device=prompt_embeds.device,
                    dtype=prompt_embeds.dtype,
                    requires_grad=True
                )
                attention_mask = None

        return self.connector(prompt_embeds), attention_mask


    def forward(self, conversations, device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.mllm_backbone = self.mllm_backbone.to(device)

        tokenize_func = self.get_tokenize_fn()
        tokenizer = self.get_tokenizer()
        conversations = [con.item() for con in conversations]
        caption = [con["text"] for con in conversations]
        video = [con["video"] for con in conversations]
        audio = [con["audio"] for con in conversations if "audio" in con]
        #start_time = time.time()
        inputs = tokenize_func(
            tokenizer, caption, video, audio
        )
  
        inputs = to_device(inputs,device,dtype = torch.bfloat16)
 

        prompt_embeds, attention_mask = self.encode_condition(**inputs)
        #print("prompt_embeds.shape:",prompt_embeds.shape,flush=True)

        return [prompt_embeds, torch.ones(prompt_embeds.shape[0], 1).to(device)]
