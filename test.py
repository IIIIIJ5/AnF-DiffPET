"""Evaluate a trained AnF-DiffPET checkpoint on the test lists.

Example:
    python test.py --data_dir data/lists --checkpoint results/anf_diffpet/model-100000.pt --save_dir results/test

Reports mean PSNR / SSIM against HD PET and, with --save_dir, writes the denoised slices
(<save_dir>/output) and [LD | denoised | HD] panels (<save_dir>/panel).
"""
import argparse
import os

from anfdiffpet import build_anf_diffpet
from anfdiffpet.data import PairedSliceDataset
from anfdiffpet.trainer import Trainer
from anfdiffpet.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--dataroot', default='')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--save_dir', default=None)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--sampling_steps', type=int, default=1000, help='1000 = ancestral DDPM; fewer steps = DDIM')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--no_msctr', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    test_lists = [os.path.join(args.data_dir, f'test_{name}.txt') for name in ('hd', 'ld', 'ct')]
    test_set = PairedSliceDataset(*test_lists, image_size=args.image_size, dataroot=args.dataroot)

    diffusion = build_anf_diffpet(image_size=args.image_size, sampling_timesteps=args.sampling_steps,
                                  use_msctr=not args.no_msctr)
    trainer = Trainer(diffusion, test_set, test_set, results_folder=os.path.dirname(args.checkpoint) or '.',
                      eval_batch_size=args.batch_size, num_workers=0)
    trainer.load(args.checkpoint, load_optimizer=False)
    trainer.evaluate(save_dir=args.save_dir)


if __name__ == '__main__':
    main()
