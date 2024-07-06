from fungiclef.dataset import EmbeddingMetadataDataset
from fungiclef.wrapper import FungiMetadataModule, FungiDinoModule
from fungiclef.transforms import get_transforms
import pandas as pd

from torch.utils.data import DataLoader
import lightning as L
import torch
from fungiclef.loss import FungiModelLoss, SeesawLoss
import timm

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import v2
from torch.utils.data import default_collate
from fungiclef.model import FungiMEEModel, NUM_CLASSES
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime

import numpy as np
import os

import random

# Load it to dataloader
# BASE_BATCH_SIZE = 1024
# BASE_LR = 1e-3

BATCH_SIZE = 128
EPOCHS = 50
WORKERS = 2
N_CLASSES = NUM_CLASSES

# LR = BASE_LR * BATCH_SIZE / BASE_BATCH_SIZE

RUN_PREFIX = "dino_"

# METADATA_WEIGHT = 0.5
UNKNOWN_LOSS_WEIGHT = 0.0
UNKNOWN_WEIGHT = 5
POISON_LOSS_WEIGHT = 0.15
POISON_WEIGHT = 100
SEESAW_LOSS_Q = 0.3
SCHEDULER = "cosine"


LR = 4e-5

train_df = pd.read_parquet("../train1.pq")
val_df = pd.read_parquet("../val1.pq")

# Load it as torch dataset
train_dataset = EmbeddingMetadataDataset(train_df)
valid_dataset = EmbeddingMetadataDataset(val_df)

model = FungiMEEModel(num_classes=N_CLASSES)

# Weighted sampling
train_dist = np.array(
    [train_df.class_id.value_counts().get(i, 1) for i in range(NUM_CLASSES)]
)
val_dist = np.array(
    [val_df.class_id.value_counts().get(i, 1) for i in range(NUM_CLASSES)]
)

train_dist = train_dist / train_dist.sum()
val_dist = val_dist / val_dist.sum()

dist_weights = val_dist / train_dist

_train_weights = [dist_weights[class_id] for class_id in train_df["class_id"]]

sampler = WeightedRandomSampler(_train_weights, len(_train_weights), replacement=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=WORKERS,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
)

run_name = RUN_PREFIX + datetime.now().strftime("%m%d%H%M")
# Use our wrapper module to get a PyTorch Lightning trainer
wandb_logger = WandbLogger(name=run_name, log_model=False, project="FungiClef")

os.makedirs("../checkpoints/" + run_name, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints/" + run_name,
    save_top_k=3,
    monitor="valid_acc_cls",
    mode="max",
    save_weights_only=True,
)
lr_monitor = LearningRateMonitor()

trainer = L.Trainer(
    callbacks=[checkpoint_callback], logger=wandb_logger, max_epochs=EPOCHS
)

# Class distribution for seesaw / CE loss
class_distribution = []
c = val_df.class_id.value_counts()
for i in range(N_CLASSES):
    class_distribution.append(c.get(i, 1))

# Try use focal loss
# loss = FungiModelLoss(loss="cross_entropy", class_distribution=class_distribution)
loss = SeesawLoss(
    class_distribution=torch.Tensor(class_distribution), num_classes=NUM_CLASSES, q=SEESAW_LOSS_Q, p=1-SEESAW_LOSS_Q
)
module = FungiDinoModule(

    model,
    learning_rate=LR,
    loss=loss.forward,
    unknown_loss_weighting=UNKNOWN_LOSS_WEIGHT,
    unknown_weighting=UNKNOWN_WEIGHT,
    poison_loss_weighting=POISON_LOSS_WEIGHT,
    poison_weighting=POISON_WEIGHT,
    scheduler=SCHEDULER,
)

trainer.fit(module, train_loader, valid_loader)
