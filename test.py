"""Evaluate a trained AnF-DiffPET checkpoint on the test lists (module switches are read from config.py).

    python test.py --checkpoint results/anf_diffpet/model-100000.pt --save_dir results/test

Reports mean PSNR / SSIM against HD PET. With --save_dir it also writes the denoised slices
(<save_dir>/output) and [LD | denoised | HD] panels (<save_dir>/panel).
测试时的模块开关必须与训练时一致（都从 config.py 读取）。
"""
import argparse

import config as cfg
from anfdiffpet.trainer import Trainer
from anfdiffpet.utils import set_seed
from train import build_from_config, dataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', default=cfg.data_dir)
    p.add_argument('--save_dir', default=None)
    p.add_argument('--sampling_steps', type=int, default=cfg.test_sampling_steps)
    args = p.parse_args()

    set_seed(cfg.seed)
    test_set = dataset(args.data_dir, 'test')
    trainer = Trainer(build_from_config(args.sampling_steps), test_set, test_set, results_folder='.',
                      eval_batch_size=cfg.test_batch_size, num_workers=0)
    trainer.load(args.checkpoint, load_optimizer=False)
    trainer.evaluate(save_dir=args.save_dir)


if __name__ == '__main__':
    main()
