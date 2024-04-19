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


# TODO: Repurpose this train script to ingest configs and stuff
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_PARQUET = "dev_train.parquet"
VAL_PARQUET = "dev_val.parquet"

train_df = pd.read_parquet(TRAIN_PARQUET)
print(len(train_df))

val_df = pd.read_parquet(VAL_PARQUET)
print(len(val_df))

