import gc
import os
import cv2
import sys
import json
import time
import timm
import torch
import random
import sklearn.metrics

from PIL import Image
from pathlib import Path
from functools import partial
from contextlib import contextmanager

import numpy as np
import scipy as sp
import pandas as pd
import torch.nn as nn

from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import torchvision

from fungiclef.model.utils import get_timm_model

MODEL_NAME = 'vit_large_patch16_224'

# TODO: Make some crazy multi-head thing. This will do for now

def classifier_model(n_classes:int , pretrained_path=None, use_imagenet_pretrain=False):
    # Make simple classifier model 
    pretrained_weights = None
    
    if pretrained_path: 
        pretrained_weights = torch.load(pretrained_path)
        use_imagenet_pretrain = False

    model = get_timm_model(MODEL_NAME, n_classes, pretrained=use_imagenet_pretrain)

    if pretrained_weights: 
        model.load_state_dict(pretrained_weights)

    return model

