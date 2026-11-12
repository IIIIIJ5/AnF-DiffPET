"""CT-conditioned denoising U-Net of AnF-DiffPET.

Inputs of one U-Net call:
    x            (B, 2, H, W)  noisy HD PET state x_t concatenated with the LD PET image (both in [-1, 1])
    time         (B,)          diffusion time step
    ct           (B, 1, H, W)  registered CT slice in [0, 1]
Output: the denoised HD PET estimate x0_hat (B, 1, H, W) in [-1, 1] (x0-prediction).

Components:
    AFG   - anatomical-frequency guidance: frozen MedDINOv3 CT features are fused into encoder stages 1 and 2
            by two cascaded frequency cross-attention (FCA) blocks.
    MSCTR - multi-scale cross-transformer reconstruction: two channel-wise cross-transformers rebuild the
            eight encoder skip features jointly before they reach the decoder.
"""
import torch
from torch import nn

from .modules.blocks import (Attention, Downsample, LinearAttention, PreNorm, PreNormPair, Residual,
                             ResnetBlock, SinusoidalPosEmb, Upsample)
from .modules.cct import ChannelTransformer
from .modules.fca import FrequencyCrossAttention
from .modules.meddinov3 import MedDINOv3Encoder

GUIDED_STAGES = (1, 2)              # encoder stages that receive CT guidance
CT_FEATURE_DIMS = {1: 96, 2: 192}   # MedDINOv3 feature channels at 1/4 and 1/8 resolution


class EncoderStage(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, groups, is_last, guide_dim=None):
        super().__init__()
        self.block1 = ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=groups)
        self.block2 = ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=groups)
        self.attn = Residual(PreNorm(dim_in, LinearAttention(dim_in)))
        self.downsample = Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)

        self.guided = guide_dim is not None
        if self.guided:
            # AFG: FCA #1 updates the CT feature with PET context (query = CT) ...
            self.fca_ct = Residual(PreNormPair(guide_dim, dim_out,
                                               FrequencyCrossAttention(guide_dim, dim_out, query_from_pet=False)))
            # ... FCA #2 guides the PET feature with the anatomy-updated feature (query = PET).
            self.fca_pet = Residual(PreNormPair(dim_out, guide_dim,
                                                FrequencyCrossAttention(guide_dim, dim_out, query_from_pet=True)))

    def forward(self, x, t, skips, ct_feat=None):
        x = self.block1(x, t)
        skips.append(x)
        x = self.attn(self.block2(x, t))
        skips.append(x)
        x = self.downsample(x)
        if self.guided:
            anatomy_updated = self.fca_ct(ct_feat, x)
            x = self.fca_pet(x, anatomy_updated)
        return x


class DecoderStage(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, groups, is_last):
        super().__init__()
        self.block1 = ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=groups)
        self.block2 = ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=groups)
        self.attn = Residual(PreNorm(dim_out, LinearAttention(dim_out)))
        self.upsample = Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)

    def forward(self, x, t, skips):
        x = self.block1(torch.cat((x, skips.pop()), dim=1), t)
        x = self.block2(torch.cat((x, skips.pop()), dim=1), t)
        return self.upsample(self.attn(x))


class AnFUNet(nn.Module):
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), channels=1, image_size=128, resnet_block_groups=8,
                 use_msctr=True, ct_backbone=None):
        super().__init__()
        self.channels = channels
        self.out_dim = channels
        self.init_conv = nn.Conv2d(channels * 2, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        self.downs = nn.ModuleList([
            EncoderStage(dim_in, dim_out, time_dim, resnet_block_groups, is_last=ind == len(in_out) - 1,
                         guide_dim=CT_FEATURE_DIMS.get(ind))
            for ind, (dim_in, dim_out) in enumerate(in_out)])

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        self.ups = nn.ModuleList([
            DecoderStage(dim_in, dim_out, time_dim, resnet_block_groups, is_last=ind == len(in_out) - 1)
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out))])

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        self.extract_ct_backbone = MedDINOv3Encoder(backbone=ct_backbone)

        # MSCTR: one cross-transformer for the first skip of every stage, one for the second.
        self.use_msctr = use_msctr
        if use_msctr:
            skip_dims = [dim_in for dim_in, _ in in_out]
            self.ctrans_block1 = ChannelTransformer(img_size=image_size, channel_num=skip_dims)
            self.ctrans_block2 = ChannelTransformer(img_size=image_size, channel_num=skip_dims)

    def forward(self, x, time, ct):
        ct_feats = self.extract_ct_backbone(ct)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        skips = []
        for ind, stage in enumerate(self.downs):
            x = stage(x, t, skips, ct_feat=ct_feats[ind])

        if self.use_msctr:
            first, second = self.ctrans_block1(*skips[0::2]), self.ctrans_block2(*skips[1::2])
            skips = [s for pair in zip(first, second) for s in pair]

        x = self.mid_block2(self.mid_attn(self.mid_block1(x, t)), t)

        for stage in self.ups:
            x = stage(x, t, skips)

        x = self.final_res_block(torch.cat((x, r), dim=1), t)
        return self.final_conv(x)
