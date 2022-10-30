import glob
import os
import numpy as np
import torch.utils.data as data
from PIL import Image


def refuge_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    data_type = 'float32' if normalized else 'uint8'
    c_map = np.zeros((N, 3), dtype=data_type)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        c_map[i] = np.array([r, g, b])

    c_map = c_map / 255 if normalized else c_map
    return c_map


class REFUGESegmentation(data.Dataset):
    c_map = refuge_cmap()

    def __init__(self, data_path, transform=None, test=False):
        self.transform = transform
        self.test = test
        self.images = glob.glob(os.path.join(data_path, '*[0-9].png'))
        self.images.sort()

        if not self.test:
            self.labels = glob.glob(os.path.join(data_path, '*_mask.png'))
            self.labels.sort()

    def __getitem__(self, index):

        if self.test:
            path_x = self.images[index]

            img_x = Image.open(path_x).convert('RGB')

            img_x = np.array(img_x)

            if self.transform is not None:
                aug = self.transform(image=img_x)
                img_x = aug['image']

            img_name = path_x.split('/')[-1].split('.')[0]

            return img_x, img_name
        else:
            path_x = self.images[index]
            path_y = self.labels[index]

            img_x = Image.open(path_x).convert('RGB')
            img_y = Image.open(path_y)

            img_x = np.array(img_x)
            img_y = np.array(img_y)

            img_y[img_y == 0] = 2
            img_y[img_y == 128] = 1
            img_y[img_y == 255] = 0

            if self.transform is not None:
                aug = self.transform(image=img_x, mask=img_y)
                img_x = aug['image']
                img_y = aug['mask']

            img_name = path_x.split('/')[-1].split('.')[0]

            return img_x, img_y, img_name

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.c_map[mask]
