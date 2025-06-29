# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from timm.models.vision_transformer import Mlp
from models.patchify import PatchEmbed1D, PatchEmbed2D
from models.attention import VanillaAttention
from models.pos_emb import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from models.WideAndDeep import WideAndDeep


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = VanillaAttention(hidden_size, num_heads, hidden_size // num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Traj_Transformer(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        traj_length=200,
        patch_size=1,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        lon_lat_embedding=True
    ):
        super().__init__()
        self.out_channels = 1 if lon_lat_embedding else 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.traj_length = traj_length
        self.lon_lat_embedding = lon_lat_embedding

        if self.lon_lat_embedding:
            print('Using 2D Patch Embedding for Trajectory Transformer')
            self.x_embedder = PatchEmbed2D(embed_dim=hidden_size)
        else:
            print('Using 1D Patch Embedding for Trajectory Transformer')
            self.x_embedder = PatchEmbed1D(embed_dim=hidden_size)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = WideAndDeep(embedding_dim=hidden_size)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.out_length, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = None
        if self.lon_lat_embedding:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.traj_length)
        else:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.x_embedder.out_length, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if not self.lon_lat_embedding:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        else:
            w_lon = self.x_embedder.proj_lon.weight.data
            nn.init.xavier_uniform_(w_lon.view([w_lon.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj_lon.bias, 0)
            w_lat = self.x_embedder.proj_lat.weight.data
            nn.init.xavier_uniform_(w_lat.view([w_lat.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj_lat.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.wide_fc.weight, std=0.02)
        nn.init.normal_(self.y_embedder.deep_fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.deep_fc2.weight, std=0.02)
        nn.init.constant_(self.y_embedder.wide_fc.bias, 0)
        nn.init.constant_(self.y_embedder.deep_fc1.bias, 0)
        nn.init.constant_(self.y_embedder.deep_fc2.bias, 0)
        nn.init.normal_(self.y_embedder.depature_embedding.weight, std=0.02)
        nn.init.normal_(self.y_embedder.sid_embedding.weight, std=0.02)
        nn.init.normal_(self.y_embedder.eid_embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if not self.spatial_temporal_attention:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
                nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        trajs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        assert x.shape[1] == self.x_embedder.out_length

        if not self.lon_lat_embedding:
            x = x.reshape(shape=(x.shape[0], self.x_embedder.out_length, p, c))
            x = torch.einsum('nlpc->nclp', x)
            trajs = x.reshape(shape=(x.shape[0], c, self.x_embedder.out_length * p))
        else:
            x_lon, x_lat = x[:, 0::2, :], x[:, 1::2, :]
            trajs = torch.cat((x_lon, x_lat), dim=2)  # (B, L, C)
            trajs = rearrange(trajs, 'B L C -> B C L').contiguous()
        return trajs

    def forward(self, x, t, y, cond_drop_prob=0.0):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        B, C, L = x.shape
        assert C == 2
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, cond_drop_prob)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
