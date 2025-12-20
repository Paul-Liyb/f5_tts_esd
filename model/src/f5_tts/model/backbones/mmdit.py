"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    MMDiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

# === 新增：Emotion Embedding ===
class EmotionEmbedding(nn.Module):
    def __init__(self, num_embeds, dim):
        super().__init__()
        # +1 用于处理 drop_emotion (CFG) 或者 padding
        self.embed = nn.Embedding(num_embeds + 1, dim)
        nn.init.zeros_(self.embed.weight)

    def forward(self, idx, drop_emotion=False):
        # 假设 idx 是 emotion IDs
        if drop_emotion:
            # 如果做 CFG (Classifier Free Guidance)，丢弃条件时返回 0 向量
            return torch.zeros_like(self.embed(idx))
        return self.embed(idx)
# text embedding


class TextEmbedding(nn.Module):
    def __init__(self, out_dim, text_num_embeds, mask_padding=True):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)  # will use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        self.precompute_max_pos = 1024
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: int["b nt"], drop_text=False) -> int["b nt d"]:
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b nt -> b nt d

        # sinus pos emb
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long)
        batch_text_len = text.shape[1]
        pos_idx = get_pos_embed_indices(batch_start, batch_text_len, max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]

        text = text + text_pos_embed

        if self.mask_padding:
            text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)

        return text


# noised input & masked cond audio embedding


class AudioEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = torch.cat((x, cond), dim=-1)
        x = self.linear(x)
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using MM-DiT blocks


class MMDiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_mask_padding=True,
        qk_norm=None,
        emotion_num_embeds=6, # 默认 5 个情感 + 1 个预留
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(dim, text_num_embeds, mask_padding=text_mask_padding)
        self.text_cond, self.text_uncond = None, None  # text cache
        self.audio_embed = AudioEmbedding(mel_dim, dim)
        self.emotion_embed = EmotionEmbedding(emotion_num_embeds, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in MMDiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
    ):
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, drop_text=True)
                c = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, drop_text=False)
                c = self.text_cond
        else:
            c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        return x, c

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        emotion: int["b"] | None = None,  # emotion id
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        drop_emotion: bool = False,  # cfg for emotion
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning (time), c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if emotion is not None:
            e = self.emotion_embed(emotion, drop_emotion=drop_emotion)
            t = t + e

        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond, c_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond, c_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t_base = self.time_embed(time)
            if emotion is not None:
                e_cond = self.emotion_embed(emotion, drop_emotion=False)
                e_uncond = self.emotion_embed(emotion, drop_emotion=True)
                t_cound = t_base + e_cond
                t_uncond = t_base + e_uncond
            else:
                t_cound = t_base
                t_uncond = t_base

            t = torch.cat((t_cound, t_uncond), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x, c = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache
            )

        seq_len = x.shape[1]
        text_len = text.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        for block in self.transformer_blocks:
            c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
