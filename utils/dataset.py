from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F

import torch
import numpy as np
import cv2 as cv
import os


class TransSet:
    """
    A transform set include random crop, random horizontal flip, random color jitter.
    """
    def __init__(self, crop=(32, 32), flip_p=0.5, color_jitter=(0.1, 0.1, 0.1, 0.1)):
        self.crop = crop
        self.flip = flip_p
        self.colo_jitter = color_jitter

    def __call__(self, img, mask):
        img, mask = transforms.F.to_pil_image(img), transforms.F.to_pil_image(mask)

        if self.crop:
            xyxy = transforms.RandomCrop.get_params(img, self.crop)
            img, mask = transforms.F.crop(img, *xyxy), transforms.F.crop(mask, *xyxy)
        if np.random.rand() > self.flip:
            img, mask = transforms.F.hflip(img), transforms.F.hflip(mask)
        if self.colo_jitter:
            img = transforms.ColorJitter(*self.colo_jitter)(img)

        return transforms.F.to_tensor(img), torch.from_numpy(np.array(mask, np.float32))


class SimpleDataset(Dataset):
    """
    A simple dataset, the data dir form should be like this:
    - data_dir_name
        + imgs
        + masks
    """
    def __init__(self, data_dir, output_channel=1, trans_set=None):
        super(SimpleDataset, self).__init__()
        self.trans_set = trans_set
        self.ids = []
        self.output_channel = output_channel

        for fn in os.listdir(data_dir + '/imgs'):
            img_path = ''.join([data_dir, '/imgs/', fn])
            mask_path = ''.join([data_dir, '/masks/', fn])
            if os.path.exists(mask_path):
                self.ids.append((img_path, mask_path))

    def __getitem__(self, index):
        img_path, mask_path = self.ids[index]

        image = cv.imread(img_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if self.trans_set:
            image, mask = self.trans_set(image, mask)
        else:
            transforms.F.to_tensor(image), torch.from_numpy(np.array(mask, np.float32))

        if self.output_channel > 1:
            mask = F.one_hot(mask, num_classes=self.output_channel).permute(2,0,1)
        else:
            mask[mask > 0] = 1
            mask = mask.view(1, *mask.shape)
        return image, mask

    def __len__(self):
        return len(self.ids)

