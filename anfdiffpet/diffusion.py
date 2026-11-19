"""Conditional diffusion model with x0-prediction for LD PET denoising.

Training: the HD PET image x0 is noised with a linear beta schedule,
    q(x_t | x0) = N(sqrt(abar_t) x0, (1 - abar_t) I),
and the U-Net predicts the clean HD PET image from (x_t, LD PET, CT, t).
Inference: deterministic DDIM sampling (eta = 0) with 10 reverse steps, starting from Gaussian noise;
only LD PET and CT are needed.
条件扩散模型（x0 预测）：训练时对 HD PET 加噪，网络直接预测干净的 HD PET；推理时从高斯噪声出发，用 10 步 DDIM 去噪。
"""
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .losses import FrequencyContrastiveHardMining
from .utils import extract, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, *, image_size, timesteps=1000, sampling_timesteps=10,
                 fchm_weight=1.0, fchm_hard_negatives=2):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        assert 1 <= sampling_timesteps <= timesteps

        self.fchm_weight = fchm_weight
        self.fchm = FrequencyContrastiveHardMining(hard_neg_num=fchm_hard_negatives) if fchm_weight > 0 else None

        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    # ----------------------------------------------------------------------------- forward process
    def q_sample(self, x0, t, noise):
        # 前向加噪：x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * noise
        return (extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def predict_x0(self, x_t, t, x_ld, ct, clip=True):
        x0_hat = self.model(torch.cat((x_t, x_ld), dim=1), t, ct)
        return x0_hat.clamp(-1., 1.) if clip else x0_hat

    # ----------------------------------------------------------------------------- reverse process
    @torch.no_grad()
    def ddim_sample(self, x_ld, ct):
        """Deterministic DDIM sampling (eta = 0) with self.sampling_timesteps reverse steps (paper: 10)."""
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        img = torch.randn_like(x_ld)
        x0_hat = img
        for time, time_next in tqdm(list(zip(times[:-1], times[1:])), desc='sampling', leave=False):
            batched_t = torch.full((img.shape[0],), time, device=img.device, dtype=torch.long)
            x0_hat = self.predict_x0(img, batched_t, x_ld, ct)
            if time_next < 0:
                break
            # 由预测的 x0 反推噪声，再确定性地跳到下一个时间步
            eps = (img - self.sqrt_alphas_cumprod[time] * x0_hat) / self.sqrt_one_minus_alphas_cumprod[time]
            img = self.sqrt_alphas_cumprod[time_next] * x0_hat + self.sqrt_one_minus_alphas_cumprod[time_next] * eps
        return x0_hat

    @torch.no_grad()
    def sample(self, x_ld, ct):
        """x_ld: LD PET in [0, 1]; ct: CT in [0, 1]. Returns the denoised HD PET estimate in [0, 1]."""
        x_ld = normalize_to_neg_one_to_one(x_ld)
        x0 = self.ddim_sample(x_ld, ct)
        return unnormalize_to_zero_to_one(x0)

    # ----------------------------------------------------------------------------- training
    def forward(self, x_hd, x_ld, ct):
        """All inputs in [0, 1]; HD and LD PET are mapped to [-1, 1], CT is used as is.

        L_total = L1(x0_hat, x_HD) + lambda_FCHM * L_FCHM(x0_hat, x_HD, LD PET of the batch)
        """
        x_hd, x_ld = normalize_to_neg_one_to_one(x_hd), normalize_to_neg_one_to_one(x_ld)
        t = torch.randint(0, self.num_timesteps, (x_hd.shape[0],), device=x_hd.device).long()  # 每个样本随机时间步
        x_t = self.q_sample(x_hd, t, torch.randn_like(x_hd))

        x0_hat = self.predict_x0(x_t, t, x_ld, ct, clip=False)  # 网络直接预测 HD PET
        loss = F.l1_loss(x0_hat, x_hd)
        if self.fchm is not None:  # 锚点：预测的 HD PET；正样本：HD PET；负样本：batch 内的 LD PET
            loss = loss + self.fchm_weight * self.fchm(x0_hat, x_hd, x_ld)
        return loss
