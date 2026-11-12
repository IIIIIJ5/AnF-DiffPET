"""Conditional DDPM with x0-prediction for LD PET denoising.

Forward process on the HD PET image x0 (1000-step linear beta schedule):
    q(x_t | x0) = N(sqrt(abar_t) x0, (1 - abar_t) I)
The U-Net predicts the clean HD PET image from (x_t, LD PET, CT, t). The reverse step uses
    mu = sqrt(abar_{t-1}) beta_t / (1 - abar_t) * x0_hat + sqrt(alpha_t) (1 - abar_{t-1}) / (1 - abar_t) * x_t
with the fixed posterior variance beta_tilde_t. Sampling starts from Gaussian noise; only LD PET and CT are needed.
条件 DDPM（x0 预测）：训练时对 HD PET 加噪，网络直接预测干净的 HD PET；推理时从高斯噪声出发逐步去噪。
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
    def __init__(self, model, *, image_size, timesteps=1000, sampling_timesteps=None,
                 fchm_weight=1.0, fchm_hard_negatives=2):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps or timesteps
        assert self.sampling_timesteps <= timesteps

        self.fchm_weight = fchm_weight
        self.fchm = FrequencyContrastiveHardMining(hard_neg_num=fchm_hard_negatives) if fchm_weight > 0 else None

        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

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
    def p_sample_loop(self, x_ld, ct):
        """Ancestral DDPM sampling over all time steps."""
        img = torch.randn_like(x_ld)  # 从纯高斯噪声出发
        for t in tqdm(reversed(range(self.num_timesteps)), desc='sampling', total=self.num_timesteps, leave=False):
            batched_t = torch.full((img.shape[0],), t, device=img.device, dtype=torch.long)
            x0_hat = self.predict_x0(img, batched_t, x_ld, ct)
            # 后验均值：由预测的 x0 与当前 x_t 线性组合得到
            mean = (extract(self.posterior_mean_coef1, batched_t, img.shape) * x0_hat +
                    extract(self.posterior_mean_coef2, batched_t, img.shape) * img)
            if t > 0:
                log_var = extract(self.posterior_log_variance_clipped, batched_t, img.shape)
                img = mean + (0.5 * log_var).exp() * torch.randn_like(img)
            else:
                img = mean
        return img

    @torch.no_grad()
    def ddim_sample(self, x_ld, ct):
        """Deterministic DDIM sampling (eta = 0) with fewer steps, for faster evaluation."""
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
        if self.sampling_timesteps < self.num_timesteps:
            x0 = self.ddim_sample(x_ld, ct)
        else:
            x0 = self.p_sample_loop(x_ld, ct)
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
