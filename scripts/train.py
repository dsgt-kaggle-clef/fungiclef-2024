from fungiclef.dataset import ImageMetadataDataset
from fungiclef.wrapper import FungiMetadataModule
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
from fungiclef.model import FungiMetadataModel, NUM_CLASSES
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np

import random

# Load it to dataloader
BATCH_SIZE = 24
EPOCHS = 50
WORKERS = 8
N_CLASSES = NUM_CLASSES
LR = 4e-5

CUTMIX_MIXUP_P = 0.25

MINI = False

if MINI:

    train_df = pd.read_csv("../train_mini.csv")
    val_df = pd.read_csv("../val_mini.csv")

    # N_CLASSES = 30

else:

    train_df = pd.read_parquet("../train.pq")[:1000]
    val_df = pd.read_parquet("../val.pq")[:1000]

# Load it as torch dataset
train_dataset = ImageMetadataDataset(
    train_df,
    local_filepath="../data/DF_FULL/",
    transform=get_transforms(data="train"),
)
valid_dataset = ImageMetadataDataset(
    val_df, local_filepath="../data/DF_FULL/", transform=get_transforms(data="valid")
)

# cutmix = v2.CutMix(num_classes=N_CLASSES)
# mixup = v2.MixUp(num_classes=N_CLASSES)
# cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
# identity = v2.Identity()
erase = v2.RandomErasing(p=CUTMIX_MIXUP_P)


def collate_fn(batch):
    return erase(*default_collate(batch))
    # return cutmix_or_mixup(*default_collate(batch))


model = FungiMetadataModel(use_metadata=True)

# Weighted sampling
train_dist = np.array([train_df.class_id.value_counts().get(i, 1) for i in range(NUM_CLASSES)])
val_dist = np.array([val_df.class_id.value_counts().get(i, 1) for i in range(NUM_CLASSES)])

train_dist = train_dist / train_dist.sum()
val_dist = val_dist / val_dist.sum()

dist_weights = val_dist / train_dist

_train_weights = [dist_weights[class_id] for class_id in train_df['class_id']]

sampler = WeightedRandomSampler(_train_weights, len(_train_weights), replacement=True)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=WORKERS,
    collate_fn=collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
)


# Use our wrapper module to get a PyTorch Lightning trainer
wandb_logger = WandbLogger(log_model=False, project="FungiClef")

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints", save_top_k=3, monitor="valid_acc_cls", mode="max", save_weights_only=True
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
    class_distribution=torch.Tensor(class_distribution), num_classes=NUM_CLASSES
)
# module = FungiMetadataModule(
#     model,
#     loss=loss.forward,
#     optimizer=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05),
# )


module = FungiDinoModule(

    model,
    learning_rate=LR,
    loss=loss.forward,
    unknown_loss_weighting=UNKNOWN_LOSS_WEIGHT,
    unknown_weighting=UNKNOWN_WEIGHT,
    poison_loss_weighting=POISON_LOSS_WEIGHT,
    poison_weighting=POISON_WEIGHT,
)


trainer.fit(module, train_loader, valid_loader)
