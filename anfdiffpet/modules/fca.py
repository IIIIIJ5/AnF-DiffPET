"""Frequency Cross-Attention (FCA), the building block of Anatomical-Frequency Guidance (AFG).

Query and key are multiplied in the Fourier domain (patch-wise FFT, as in FFTformer),
and the resulting map modulates the spatial-domain value branch.
"""
import torch
from einops import rearrange
from torch import nn


class FrequencyCrossAttention(nn.Module):
    """Cross-attention between a PET feature map and a CT-derived guidance feature map.

    Args:
        dim_guide: channels of the guidance feature (CT anatomy or anatomy-updated feature).
        dim_pet: channels of the PET feature from the denoising U-Net.
        query_from_pet: if True, the query comes from the PET feature and the key/value from the guidance
            feature (output has `dim_pet` channels); otherwise the roles are swapped (output has `dim_guide` channels).
    """

    def __init__(self, dim_guide, dim_pet, query_from_pet=True, bias=False, patch_size=4):
        super().__init__()
        dim_q, dim_kv = (dim_pet, dim_guide) if query_from_pet else (dim_guide, dim_pet)

        self.to_hidden_q = nn.Conv2d(dim_q, dim_q, kernel_size=1, bias=bias)
        self.to_hidden_kv = nn.Conv2d(dim_kv, dim_q * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw_q = nn.Conv2d(dim_q, dim_q, kernel_size=3, stride=1, padding=1, groups=dim_q, bias=bias)
        self.to_hidden_dw_kv = nn.Conv2d(dim_q * 2, dim_q * 2, kernel_size=3, stride=1, padding=1,
                                         groups=dim_q * 2, bias=bias)
        self.project_out = nn.Conv2d(dim_q, dim_q, kernel_size=1, bias=bias)
        self.patch_size = patch_size

    def forward(self, x_query, x_kv):
        q = self.to_hidden_dw_q(self.to_hidden_q(x_query))
        k, v = self.to_hidden_dw_kv(self.to_hidden_kv(x_kv)).chunk(2, dim=1)

        p = self.patch_size
        q_patch = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)
        k_patch = rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        attn = torch.fft.irfft2(q_fft * k_fft, s=(p, p))
        attn = rearrange(attn, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=p, p2=p)

        return self.project_out(v * attn)
