from .diffusion import GaussianDiffusion
from .unet import AnFUNet

__all__ = ['GaussianDiffusion', 'AnFUNet', 'build_anf_diffpet']


def build_anf_diffpet(image_size=128, dim=64, dim_mults=(1, 2, 4, 8), sampling_timesteps=1000,
                      use_msctr=True, use_fchm=True, fchm_weight=1.0, fchm_k=2, ct_backbone=None):
    """Full AnF-DiffPET model: x0-predicting U-Net with AFG + MSCTR, trained with L1 + FCHM.

    sampling_timesteps = 1000 gives ancestral DDPM sampling; fewer steps switch to DDIM.
    """
    unet = AnFUNet(dim=dim, dim_mults=dim_mults, channels=1, image_size=image_size,
                   use_msctr=use_msctr, ct_backbone=ct_backbone)
    return GaussianDiffusion(unet, image_size=image_size, timesteps=1000, sampling_timesteps=sampling_timesteps,
                             fchm_weight=fchm_weight if use_fchm else 0., fchm_hard_negatives=fchm_k)
