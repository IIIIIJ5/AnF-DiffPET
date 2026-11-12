"""Frequency-Contrastive Hard Mining (FCHM), a training-only frequency-domain regulariser."""
import torch
from torch import nn


class FrequencyContrastiveHardMining(nn.Module):
    """L_FCHM = d_ap / (mean_{j in H_k} d_an^j + delta), averaged over the mini-batch.

    Anchor: predicted HD PET. Positive: paired HD PET. Negatives: the paired LD PET image and the other
    LD PET images of the mini-batch. Distances are L2 norms between FFT magnitude spectra; H_k holds the
    k negatives closest to the anchor (online hard example mining).

    Args:
        anchor:    (B, C, H, W) predicted HD PET
        positive:  (B, C, H, W) HD PET
        negatives: (B, C, H, W) LD PET of the same mini-batch
    """

    def __init__(self, hard_neg_num=2, delta=1e-6):
        super().__init__()
        self.hard_neg_num = hard_neg_num
        self.delta = delta

    def forward(self, anchor, positive, negatives):
        a = torch.abs(torch.fft.fft2(anchor, dim=(-2, -1))).flatten(1)
        p = torch.abs(torch.fft.fft2(positive, dim=(-2, -1))).flatten(1)
        n = torch.abs(torch.fft.fft2(negatives, dim=(-2, -1))).flatten(1)

        d_ap = torch.linalg.vector_norm(a - p, dim=1)                         # (B,)
        d_an = torch.cdist(a.unsqueeze(0), n.unsqueeze(0)).squeeze(0)         # (B, B): anchor i vs negative j
        k = min(self.hard_neg_num, d_an.shape[1])
        d_an_hard, _ = torch.topk(d_an, k=k, dim=1, largest=False)            # the k closest are the hardest

        return (d_ap / (d_an_hard.mean(dim=1) + self.delta)).mean()
