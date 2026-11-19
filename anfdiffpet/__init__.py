from .diffusion import GaussianDiffusion
from .unet import AnFUNet

__all__ = ['GaussianDiffusion', 'AnFUNet', 'build_anf_diffpet']


def build_anf_diffpet(image_size=128, dim=64, dim_mults=(1, 2, 4, 8), sampling_timesteps=10,
                      use_afg=True, use_msctr=True, use_fchm=True, fchm_weight=1.0, fchm_k=2, ct_backbone=None):
    """AnF-DiffPET: x0-predicting U-Net (AFG + MSCTR) in a conditional diffusion model, trained with L1 + FCHM.

    Inference uses deterministic DDIM sampling with sampling_timesteps reverse steps (paper: 10).
    use_afg / use_msctr / use_fchm are the module switches (默认全部开启，关闭即为对应消融).
    """
    unet = AnFUNet(dim=dim, dim_mults=dim_mults, channels=1, image_size=image_size,
                   use_afg=use_afg, use_msctr=use_msctr, ct_backbone=ct_backbone)
    return GaussianDiffusion(unet, image_size=image_size, timesteps=1000, sampling_timesteps=sampling_timesteps,
                             fchm_weight=fchm_weight if use_fchm else 0., fchm_hard_negatives=fchm_k)
