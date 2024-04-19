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
import torch.nn as nnPs

from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

from fungiclef.logger import init_logger
import wandb

WANDB_PROJECT_NAME = "FungiClef"

# TODO: Repurpose this train script to ingest configs and stuff
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_PARQUET = "dev_train.parquet"
VAL_PARQUET = "dev_val.parquet"

train_df = pd.read_parquet(TRAIN_PARQUET)
val_df = pd.read_parquet(VAL_PARQUET)

logger = init_logger()

wandb.init(
    project=WANDB_PROJECT_NAME, 
    name="vit_test", # TODO: Config name here as well
    config=None, # TODO: To add config here
)

N_CLASSES = len(train_df['class_id'].unique())

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # file_path = self.df['image_path'].values[idx]
        label = self.df['class_id'].values[idx]
        image = np.frombuffer(self.df["im_bytes"][idx], dtype="uint8").reshape(
            (self.df["im_height"][idx], self.df["im_widgth"][idx], 3)
        )
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label