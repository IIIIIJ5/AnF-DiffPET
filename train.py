"""Train AnF-DiffPET.

Example:
    python train.py --data_dir data/lists --results_dir results/anf_diffpet

`data_dir` must contain train_hd.txt, train_ld.txt, train_ct.txt, test_hd.txt, test_ld.txt and test_ct.txt
(see README.md). Training resumes automatically from <results_dir>/last.pt if it exists.
"""
import argparse
import os

from anfdiffpet import build_anf_diffpet
from anfdiffpet.data import PairedSliceDataset
from anfdiffpet.trainer import Trainer
from anfdiffpet.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True, help='folder with the six image-list files')
    p.add_argument('--dataroot', default='', help='prefix joined to every path in the lists')
    p.add_argument('--results_dir', default='results/anf_diffpet')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--grad_accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_steps', type=int, default=100000)
    p.add_argument('--sampling_steps', type=int, default=50,
                   help='DDIM steps for the periodic evaluation during training (test.py samples with full DDPM)')
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--save_last_every', type=int, default=5000, help='steps between updates of last.pt')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--no_msctr', action='store_true', help='ablation: plain U-Net skip connections')
    p.add_argument('--no_fchm', action='store_true', help='ablation: L1 loss only')
    p.add_argument('--fchm_weight', type=float, default=1.0)
    p.add_argument('--fchm_k', type=int, default=2, help='number of hard negatives')
    p.add_argument('--debug', action='store_true', help='200 steps, evaluation every 100 steps')
    return p.parse_args()


def lists(data_dir, split):
    return [os.path.join(data_dir, f'{split}_{name}.txt') for name in ('hd', 'ld', 'ct')]


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.debug:
        args.num_steps, args.eval_every = 200, 100

    train_set = PairedSliceDataset(*lists(args.data_dir, 'train'), image_size=args.image_size,
                                   dataroot=args.dataroot)
    test_set = PairedSliceDataset(*lists(args.data_dir, 'test'), image_size=args.image_size,
                                  dataroot=args.dataroot)

    diffusion = build_anf_diffpet(image_size=args.image_size, sampling_timesteps=args.sampling_steps,
                                  use_msctr=not args.no_msctr, use_fchm=not args.no_fchm,
                                  fchm_weight=args.fchm_weight, fchm_k=args.fchm_k)

    trainer = Trainer(diffusion, train_set, test_set, results_folder=args.results_dir,
                      batch_size=args.batch_size, gradient_accumulate_every=args.grad_accum, lr=args.lr,
                      num_steps=args.num_steps, eval_every=args.eval_every,
                      save_last_every=args.save_last_every, num_workers=args.num_workers)

    last = os.path.join(args.results_dir, 'last.pt')
    if os.path.exists(last):
        trainer.load(last)
    trainer.train()


if __name__ == '__main__':
    main()
