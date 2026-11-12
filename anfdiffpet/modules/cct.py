"""Channel-wise Cross-Transformer (CCT) used by Multi-Scale Cross-Transformer Reconstruction (MSCTR).

Adapted from UCTransNet (Wang et al., AAAI 2022, https://github.com/McGregorWwww/UCTransNet).
Each of the four U-Net skip features gives its own query, while keys and values come from the
concatenation of all four scales, so information is exchanged across scales before decoding.
"""
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, Dropout, LayerNorm, Softmax
from torch.nn.modules.utils import _pair


def default_cct_config(channel_num=(64, 64, 128, 256)):
    return SimpleNamespace(
        KV_size=sum(channel_num),
        expand_ratio=4,  # MLP expansion ratio
        patch_sizes=[16, 8, 4, 2],
        transformer={
            'num_heads': 4,
            'num_layers': 4,
            'embeddings_dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
            'dropout_rate': 0,
        },
    )


class Channel_Embeddings(nn.Module):
    """Patch embedding plus learned position embedding for one skip scale."""

    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer['embeddings_dropout_rate'])

    def forward(self, x):
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)  # (B, n_patches, C)
        return self.dropout(x + self.position_embeddings)


class Reconstruct(nn.Module):
    """Tokens back to a feature map at the original resolution of the scale."""

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return self.activation(self.norm(self.conv(x)))


class Attention_org(nn.Module):
    """Multi-head channel cross-attention: per-scale queries, keys/values from all scales."""

    def __init__(self, config, channel_num):
        super().__init__()
        self.KV_size = config.KV_size
        self.num_attention_heads = config.transformer['num_heads']

        self.query1, self.query2, self.query3, self.query4 = (nn.ModuleList() for _ in range(4))
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        for _ in range(self.num_attention_heads):
            self.query1.append(nn.Linear(channel_num[0], channel_num[0], bias=False))
            self.query2.append(nn.Linear(channel_num[1], channel_num[1], bias=False))
            self.query3.append(nn.Linear(channel_num[2], channel_num[2], bias=False))
            self.query4.append(nn.Linear(channel_num[3], channel_num[3], bias=False))
            self.key.append(nn.Linear(self.KV_size, self.KV_size, bias=False))
            self.value.append(nn.Linear(self.KV_size, self.KV_size, bias=False))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.attn_dropout = Dropout(config.transformer['attention_dropout_rate'])
        self.proj_dropout = Dropout(config.transformer['attention_dropout_rate'])

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        embs = (emb1, emb2, emb3, emb4)
        query_sets = (self.query1, self.query2, self.query3, self.query4)
        outs = (self.out1, self.out2, self.out3, self.out4)

        K = torch.stack([key(emb_all) for key in self.key], dim=1)            # (B, heads, N, KV)
        V = torch.stack([value(emb_all) for value in self.value], dim=1)      # (B, heads, N, KV)
        probs = []
        for emb, queries in zip(embs, query_sets):
            Q = torch.stack([query(emb) for query in queries], dim=1).transpose(-1, -2)  # (B, heads, C, N)
            scores = torch.matmul(Q, K) / math.sqrt(self.KV_size)                        # (B, heads, C, KV)
            probs.append(self.softmax(self.psi(scores)))
        probs = [self.attn_dropout(p) for p in probs]

        V = V.transpose(-1, -2)
        contexts = [torch.matmul(p, V).permute(0, 3, 2, 1).contiguous().mean(dim=3) for p in probs]  # (B, N, C)
        outputs = [out(context) for out, context in zip(outs, contexts)]
        return [self.proj_dropout(o) for o in outputs]


class Mlp(nn.Module):
    def __init__(self, config, in_channel, mlp_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer['dropout_rate'])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))


class Block_ViT(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.attn_norm = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn = Attention_org(config, channel_num)
        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.ffn1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = Mlp(config, channel_num[1], channel_num[1] * expand_ratio)
        self.ffn3 = Mlp(config, channel_num[2], channel_num[2] * expand_ratio)
        self.ffn4 = Mlp(config, channel_num[3], channel_num[3] * expand_ratio)

    def forward(self, emb1, emb2, emb3, emb4):
        emb_all = self.attn_norm(torch.cat([emb1, emb2, emb3, emb4], dim=2))
        cx = self.channel_attn(self.attn_norm1(emb1), self.attn_norm2(emb2),
                               self.attn_norm3(emb3), self.attn_norm4(emb4), emb_all)
        org = [emb1 + cx[0], emb2 + cx[1], emb3 + cx[2], emb4 + cx[3]]
        return (org[0] + self.ffn1(self.ffn_norm1(org[0])),
                org[1] + self.ffn2(self.ffn_norm2(org[1])),
                org[2] + self.ffn3(self.ffn_norm3(org[2])),
                org[3] + self.ffn4(self.ffn_norm4(org[3])))


class Encoder(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        for _ in range(config.transformer['num_layers']):
            self.layer.append(Block_ViT(config, channel_num))

    def forward(self, emb1, emb2, emb3, emb4):
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4 = layer_block(emb1, emb2, emb3, emb4)
        return (self.encoder_norm1(emb1), self.encoder_norm2(emb2),
                self.encoder_norm3(emb3), self.encoder_norm4(emb4))


class ChannelTransformer(nn.Module):
    """Reconstructs the four encoder skip features jointly; returns them with a residual connection."""

    def __init__(self, img_size, channel_num=(64, 64, 128, 256), config=None):
        super().__init__()
        config = config or default_cct_config(channel_num)
        p1, p2, p3, p4 = config.patch_sizes
        self.embeddings_1 = Channel_Embeddings(config, p1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, p2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, p3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, p4, img_size=img_size // 8, in_channels=channel_num[3])
        self.encoder = Encoder(config, channel_num)
        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(p1, p1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(p2, p2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(p3, p3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(p4, p4))

    def forward(self, en1, en2, en3, en4):
        encoded = self.encoder(self.embeddings_1(en1), self.embeddings_2(en2),
                               self.embeddings_3(en3), self.embeddings_4(en4))
        return [self.reconstruct_1(encoded[0]) + en1,
                self.reconstruct_2(encoded[1]) + en2,
                self.reconstruct_3(encoded[2]) + en3,
                self.reconstruct_4(encoded[3]) + en4]
