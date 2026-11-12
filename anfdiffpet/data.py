"""Paired HD PET / LD PET / CT slices listed in text files (one image path per line, same order in every list).
数据读取：三个列表按行一一对应，每行是一张二维灰度切片的路径。
"""
import os

import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def read_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


class PairedSliceDataset(Dataset):
    """Returns (HD PET, LD PET, CT, file name); images are single-channel tensors in [0, 1]."""

    def __init__(self, hd_list, ld_list, ct_list, image_size, dataroot=''):
        super().__init__()
        self.lists = [read_list(p) for p in (hd_list, ld_list, ct_list)]
        assert len({len(lst) for lst in self.lists}) == 1, 'HD / LD / CT lists must have the same length'
        self.dataroot = dataroot
        self.image_size = image_size

    def __len__(self):
        return len(self.lists[0])

    def _load(self, rel_path):
        # 补零到 8 的倍数（且不小于 image_size），再双三次插值缩放到 image_size x image_size
        img = np.asarray(Image.open(os.path.join(self.dataroot, rel_path)).convert('L'))
        img = pad_to_size(img, self.image_size)
        img = Image.fromarray(img).resize((self.image_size, self.image_size), Image.BICUBIC)
        return TF.to_tensor(np.array(img)).float()

    def __getitem__(self, index):
        hd, ld, ct = [self._load(lst[index]) for lst in self.lists]
        return hd, ld, ct, os.path.basename(self.lists[0][index])


def pad_to_size(img, min_size, block_size=8):
    """Zero-pads the bottom/right so that both sides are >= min_size and multiples of block_size."""
    h, w = img.shape[:2]
    target_h = int(np.ceil(max(h, min_size) / block_size) * block_size)
    target_w = int(np.ceil(max(w, min_size) / block_size) * block_size)
    return cv2.copyMakeBorder(img, 0, target_h - h, 0, target_w - w, cv2.BORDER_CONSTANT, value=0)
