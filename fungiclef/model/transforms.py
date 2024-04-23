from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop
import torch.nn as nn

WIDTH = 224
HEIGHT = 224


def get_transforms(*, data, model=None):
    assert data in ('train', 'valid')

    model_mean = list(model.default_cfg['mean']) if model else (0.5, 0.5, 0.5) 
    model_std = list(model.default_cfg['std']) if model else (0.5, 0.5, 0.5) 

    if data == 'train':
        return Compose([
            RandomResizedCrop(WIDTH, HEIGHT, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=model_mean, std=model_std),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(WIDTH, HEIGHT),
            Normalize(mean=model_mean, std=model_std),
            ToTensorV2(),
        ])