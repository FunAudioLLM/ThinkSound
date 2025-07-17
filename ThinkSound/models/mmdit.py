import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from .embeddings import compute_rope_rotations
from .embeddings import TimestepEmbedder
from .blocks import MLP, ChannelLastConv1d, ConvMLP
from .transformer_layers import (FinalBlock, JointBlock, MMDitSingleBlock)
from .utils import resample

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor


class MMmodule(nn.Module):

    def __init__(self,
                 *,
                 latent_dim: int,
                 clip_dim: int,
                 sync_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 clip_seq_len: int,
                 sync_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 v2: bool = False,
                 kernel_size: int = 7,
                 sync_kernel: int = 7,
                 use_inpaint: bool = False,
                 use_mlp: bool = False,
                 cross_attend: bool = False,
                 add_video: bool = False,
                 triple_fusion: bool = False,
                 gated_video: bool = False) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.cross_attend = cross_attend
        self.add_video = add_video
        self.gated_video = gated_video
        self.triple_fusion = triple_fusion
        self.use_inpaint = use_inpaint
        if self.gated_video:
            self.gated_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim*2, hidden_dim * 4, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
                nn.Sigmoid()
            )
            # 初始化最后一层权重为零，促进初始均匀融合
            nn.init.zeros_(self.gated_mlp[3].weight)
        if self.triple_fusion:
            self.gated_mlp_v = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim*3, hidden_dim * 4, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
                nn.Sigmoid()
            )
            self.gated_mlp_t = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim*3, hidden_dim * 4, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
                nn.Sigmoid()
            )
            nn.init.zeros_(self.gated_mlp_v[3].weight)
            nn.init.zeros_(self.gated_mlp_t[3].weight)
        if v2:
            padding_size = (kernel_size - 1) // 2
            if use_inpaint:
                self.audio_input_proj = nn.Sequential(
                    ChannelLastConv1d(latent_dim*2, hidden_dim, kernel_size=kernel_size, padding=padding_size),
                    nn.SiLU(),
                    ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=kernel_size, padding=padding_size),
                )
            else:
                self.audio_input_proj = nn.Sequential(
                    ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=kernel_size, padding=padding_size),
                    nn.SiLU(),
                    ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=kernel_size, padding=padding_size),
                )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )
            sync_pad = (sync_kernel - 1) // 2
            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=sync_kernel, padding=sync_pad),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )
        else:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )
        
        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        if use_mlp:
            self.text_cond_proj = nn.Sequential(
                nn.Linear(1024, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )
        else:
            self.text_cond_proj = nn.Linear(1024, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        if v2:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=hidden_dim,
                                            max_period=1)
        else:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=256,
                                            max_period=10000)
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                       num_heads,
                       mlp_ratio=mlp_ratio,
                       pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size, padding=padding_size, cross_attend=cross_attend)
            for i in range(fused_depth)
        ])

        if empty_string_feat is None:
            empty_string_feat = torch.zeros((77, 1024))
        
        empty_t5_feat = torch.zeros((77, 2048))

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_t5_feat = nn.Parameter(empty_t5_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)

        self.initialize_weights()
        self.initialize_rotations()

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        clip_rot = compute_rope_rotations(self._clip_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * self._latent_seq_len /
                                          self._clip_seq_len,
                                          device=self.device)

        self.latent_rot = nn.Buffer(latent_rot, persistent=False)
        self.clip_rot = nn.Buffer(clip_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        # empty string feat shall be initialized by a CLIP encoder
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def preprocess_conditions(self, clip_f: torch.Tensor, sync_f: torch.Tensor,
                              text_f: torch.Tensor, t5_features: torch.Tensor, metaclip_global_text_features: torch.Tensor) -> PreprocessedConditions:
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        # breakpoint()
        assert clip_f.shape[1] == self._clip_seq_len, f'{clip_f.shape=} {self._clip_seq_len=}'
        assert sync_f.shape[1] == self._sync_seq_len, f'{sync_f.shape=} {self._sync_seq_len=}'
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = clip_f.shape[0]

        # B * num_segments (24) * 8 * 768
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)

        # extend vf to match x
        clip_f = self.clip_input_proj(clip_f)  # (B, VN, D)
        sync_f = self.sync_input_proj(sync_f)  # (B, VN, D)

        if t5_features is not None:

            if metaclip_global_text_features is not None:
                text_f_c = self.text_cond_proj(metaclip_global_text_features)  # (B, D)
            else:
                text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)
            # 计算填充长度
            padding_size = t5_features.size(2) - text_f.size(2)  # 渴望填充的数量
            # 当确实需要填充的时候，确保填充是正数
            if padding_size > 0:
                # 填充 text_f 的特征维度两侧
                text_f = F.pad(text_f, pad=(0, padding_size), mode='constant', value=0)  # 在最后一个维度上进行填充
            else:
                text_f = text_f  # 如果填充长度不是正数，则不需要填充
            text_concat = torch.cat((text_f, t5_features), dim=1)
            text_f = self.text_input_proj(text_concat)  # (B, VN, D)
        else:
            text_f = self.text_input_proj(text_f)  # (B, VN, D)
            if metaclip_global_text_features is not None:
                text_f_c = self.text_cond_proj(metaclip_global_text_features)  # (B, D)
            else:
                text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)

        # upsample the sync features to match the audio
        sync_f = sync_f.transpose(1, 2)  # (B, D, VN)
        # sync_f = resample(sync_f, self._latent_seq_len)
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode='nearest-exact')
        sync_f = sync_f.transpose(1, 2)  # (B, N, D)

        # get conditional features from the clip side
        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))  # (B, D)

        return PreprocessedConditions(clip_f=clip_f,
                                      sync_f=sync_f,
                                      text_f=text_f,
                                      clip_f_c=clip_f_c,
                                      text_f_c=text_f_c)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions, inpaint_masked_input=None, cfg_scale:float=1.0,cfg_dropout_prob:float=0.0,scale_phi:float=0.0
                     ) -> torch.Tensor:
        """
        for non-cacheable computations
        """
        # print(f'cfg_scale: {cfg_scale}, cfg_dropout_prob: {cfg_dropout_prob}, scale_phi: {scale_phi}')
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'
        empty_conditions = None
        if inpaint_masked_input is not None:
            inpaint_masked_input = inpaint_masked_input.transpose(1,2)
        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        text_f = conditions.text_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c
            
        # breakpoint()
        if inpaint_masked_input is not None:
            latent = torch.cat([latent,inpaint_masked_input],dim=2)
        latent = self.audio_input_proj(latent)  # (B, N, D)
        global_c = self.global_cond_mlp(clip_f_c + text_f_c)  # (B, D)
        # global_c = text_f_c
        #print(f"[DEBUG] t_emb.shape = {self.t_embed(t).shape}, global_c.shape = {global_c.shape}")
        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)  # (B, D)
        extended_c = global_c + sync_f

        for block in self.joint_blocks:
            latent, clip_f, text_f = block(latent, clip_f, text_f, global_c, extended_c,
                                           self.latent_rot, self.clip_rot)  # (B, N, D)
        if self.add_video:
            if clip_f.shape[1] != latent.shape[1]:
                clip_f = resample(clip_f, latent)

            if self.triple_fusion:
                text_f = torch.mean(text_f, dim=1, keepdim=True) # (bsz, 1, D)
                text_f = text_f.expand(-1,latent.shape[1], -1) # (T_audio, D)
                fusion = torch.concat((latent, clip_f, text_f),dim=-1)
                gate_v = self.gated_mlp_v(fusion)
                gate_t = self.gated_mlp_t(fusion)
                # modulated_latent = gate * latent # 非对称设计
                latent = latent + gate_v * clip_f + gate_t * text_f
            elif self.gated_video:
                fusion = torch.concat((latent, clip_f),dim=-1)
                gate = self.gated_mlp(fusion)
                modulated_latent = gate * latent # 非对称设计
                latent = latent + modulated_latent
            else:
                latent = latent + clip_f
        
        for block in self.fused_blocks:
            if self.cross_attend:
                latent = block(latent, extended_c, self.latent_rot, context=text_f)
            else:
                latent = block(latent, extended_c, self.latent_rot)

        # should be extended_c; this is a minor implementation error #55
        flow = self.final_layer(latent, extended_c)  # (B, N, out_dim), remove t
        return flow

    def forward(self, latent: torch.Tensor, t: torch.Tensor, clip_f: torch.Tensor, sync_f: torch.Tensor,
                text_f: torch.Tensor, inpaint_masked_input, t5_features, metaclip_global_text_features, cfg_scale:float,cfg_dropout_prob:float,scale_phi:float) -> torch.Tensor:
        """
        latent: (B, N, C) 
        vf: (B, T, C_V)
        t: (B,)
        """
        # breakpoint()
        # print(f'cfg_scale: {cfg_scale}, cfg_dropout_prob: {cfg_dropout_prob}, scale_phi: {scale_phi}')
        if self.use_inpaint and inpaint_masked_input is None:
            inpaint_masked_input = torch.zeros_like(latent, device=latent.device)
        latent = latent.permute(0, 2, 1)

        if cfg_dropout_prob > 0.0:
            if inpaint_masked_input is not None:
                null_embed = torch.zeros_like(inpaint_masked_input,device=latent.device)
                dropout_mask = torch.bernoulli(torch.full((inpaint_masked_input.shape[0], 1, 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
                inpaint_masked_input = torch.where(dropout_mask, null_embed, inpaint_masked_input)

            null_embed = torch.zeros_like(clip_f,device=latent.device)
            dropout_mask = torch.bernoulli(torch.full((clip_f.shape[0], 1, 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
            # clip_f = torch.where(dropout_mask, null_embed, clip_f)
            clip_f = torch.where(dropout_mask, self.empty_clip_feat, clip_f)
            null_embed = torch.zeros_like(sync_f,device=latent.device)
            dropout_mask = torch.bernoulli(torch.full((sync_f.shape[0], 1, 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
            # sync_f = torch.where(dropout_mask, null_embed, sync_f)
            sync_f = torch.where(dropout_mask, self.empty_sync_feat, sync_f)
            null_embed = torch.zeros_like(text_f,device=latent.device)
            dropout_mask = torch.bernoulli(torch.full((text_f.shape[0], 1, 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
            # text_f = torch.where(dropout_mask, null_embed, text_f)
            text_f = torch.where(dropout_mask, self.empty_string_feat, text_f)
            if t5_features is not None:
                null_embed = torch.zeros_like(t5_features,device=latent.device)
                dropout_mask = torch.bernoulli(torch.full((t5_features.shape[0], 1, 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
                # t5_features = torch.where(dropout_mask, null_embed, t5_features)
                t5_features = torch.where(dropout_mask, self.empty_t5_feat, t5_features)
            if metaclip_global_text_features is not None:
                null_embed = torch.zeros_like(metaclip_global_text_features,device=latent.device)
                dropout_mask = torch.bernoulli(torch.full((metaclip_global_text_features.shape[0], 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
                metaclip_global_text_features = torch.where(dropout_mask, null_embed, metaclip_global_text_features)
            # null_embed = torch.zeros_like(clip_f_c,device=latent.device)
            # dropout_mask = torch.bernoulli(torch.full((clip_f_c.shape[0], 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
            # clip_f_c = torch.where(dropout_mask, null_embed, clip_f_c)
            # null_embed = torch.zeros_like(text_f_c,device=latent.device)
            # dropout_mask = torch.bernoulli(torch.full((text_f_c.shape[0], 1), cfg_dropout_prob, device=latent.device)).to(torch.bool)
            # text_f_c = torch.where(dropout_mask, null_embed, text_f_c)

        if cfg_scale != 1.0:
            # empty_conditions = self.get_empty_conditions(latent.shape[0])
            # breakpoint()
            bsz = latent.shape[0]
            latent = torch.cat([latent,latent], dim=0)
            if inpaint_masked_input is not None:
                empty_inpaint_masked_input = torch.zeros_like(inpaint_masked_input, device=latent.device)
                inpaint_masked_input = torch.cat([inpaint_masked_input,empty_inpaint_masked_input], dim=0)
            t = torch.cat([t, t], dim=0)
            empty_clip_f = torch.zeros_like(clip_f, device=latent.device)
            empty_sync_f = torch.zeros_like(sync_f, device=latent.device)
            empty_text_f = torch.zeros_like(text_f, device=latent.device)

            # clip_f = torch.cat([clip_f,empty_clip_f], dim=0)
            # sync_f = torch.cat([sync_f,empty_sync_f], dim=0)
            # text_f = torch.cat([text_f,empty_text_f], dim=0)
            clip_f = safe_cat(clip_f,self.get_empty_clip_sequence(bsz), dim=0, match_dim=1)
            sync_f = safe_cat(sync_f,self.get_empty_sync_sequence(bsz), dim=0, match_dim=1)
            text_f = safe_cat(text_f,self.get_empty_string_sequence(bsz), dim=0, match_dim=1)
            if t5_features is not None:
                empty_t5_features = torch.zeros_like(t5_features, device=latent.device)
                # t5_features = torch.cat([t5_features,empty_t5_features], dim=0)
                t5_features = torch.cat([t5_features,self.get_empty_t5_sequence(bsz)], dim=0)
            if metaclip_global_text_features is not None:
                empty_metaclip_global_text_features = torch.zeros_like(metaclip_global_text_features, device=latent.device)
                metaclip_global_text_features = torch.cat([metaclip_global_text_features,empty_metaclip_global_text_features], dim=0)
            # metaclip_global_text_features = torch.cat([metaclip_global_text_features,metaclip_global_text_features], dim=0)
            # clip_f_c = torch.cat([clip_f_c,empty_clip_f_c], dim=0)
            # text_f_c = torch.cat([text_f_c,empty_text_f_c], dim=0)

        conditions = self.preprocess_conditions(clip_f, sync_f, text_f, t5_features, metaclip_global_text_features)
        flow = self.predict_flow(latent, t, conditions, inpaint_masked_input, cfg_scale,cfg_dropout_prob,scale_phi)
        if cfg_scale != 1.0:
            cond_output, uncond_output = torch.chunk(flow, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                flow = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                flow = cfg_output
        flow = flow.permute(0, 2, 1)
        return flow

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_t5_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_t5_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)

        empty_clip = self.get_empty_clip_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)
        conditions = self.preprocess_conditions(empty_clip, empty_sync, empty_text)
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def load_weights(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'clip_rot' in src_dict:
            del src_dict['clip_rot']

        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.empty_clip_feat.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len

















def truncate_to_target(tensor, target_size, dim=1):
    current_size = tensor.size(dim)
    if current_size > target_size:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_size)
        return tensor[slices]
    return tensor

def pad_to_target(tensor, target_size, dim=1, pad_value=0):
    current_size = tensor.size(dim)
    if current_size < target_size:
        pad_size = target_size - current_size
        
        pad_config = [0, 0] * tensor.dim()
        pad_index = 2 * (tensor.dim() - dim - 1) + 1
        pad_config[pad_index] = pad_size
        
        return torch.nn.functional.pad(tensor, pad_config, value=pad_value)
    return tensor


def safe_cat(tensor1, tensor2, dim=0, match_dim=1):

    target_size = tensor2.size(match_dim)

    if tensor1.size(match_dim) > target_size:
        tensor1 = truncate_to_target(tensor1, target_size, match_dim)
        
    else:
        tensor1 = pad_to_target(tensor1, target_size, match_dim)
    
    return torch.cat([tensor1, tensor2], dim=dim)

