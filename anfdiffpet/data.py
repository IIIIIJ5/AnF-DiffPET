"""Paired HD PET / LD PET / CT slices listed in text files (one image path per line, same order in every list)."""
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
    """Returns (HD PET, LD PET, CT, file name); images are single-channel tensors in [0, 1].

    Each slice is zero-padded on the bottom/right to at least `image_size` and to a multiple of 8,
    then resized to image_size x image_size with bicubic interpolation.
    """

    def __init__(self, hd_list, ld_list, ct_list, image_size, dataroot='', augment_flip=False):
        super().__init__()
        self.hd = read_list(hd_list)
        self.ld = read_list(ld_list)
        self.ct = read_list(ct_list)
        assert len(self.hd) == len(self.ld) == len(self.ct), 'HD / LD / CT lists must have the same length'
        self.dataroot = dataroot
        self.image_size = image_size
        self.augment_flip = augment_flip

    def __len__(self):
        return len(self.ld)

    def _load(self, rel_path):
        img = np.asarray(Image.open(os.path.join(self.dataroot, rel_path)).convert('L'))
        img = pad_to_size(img, self.image_size)
        img = Image.fromarray(img).resize((self.image_size, self.image_size), Image.BICUBIC)
        return np.asarray(img)

    def __getitem__(self, index):
        images = [self._load(paths[index]) for paths in (self.hd, self.ld, self.ct)]
        if self.augment_flip:
            images = [np.ascontiguousarray(np.fliplr(img)) for img in images]
        tensors = [TF.to_tensor(img).float() for img in images]
        return tensors[0], tensors[1], tensors[2], os.path.basename(self.hd[index])


def pad_to_size(img, min_size, block_size=8):
    h, w = img.shape[:2]
    target_h = max(h, min_size)
    target_w = max(w, min_size)
    target_h = int(np.ceil(target_h / block_size) * block_size)
    target_w = int(np.ceil(target_w / block_size) * block_size)
    return cv2.copyMakeBorder(img, 0, target_h - h, 0, target_w - w, cv2.BORDER_CONSTANT, value=0)
