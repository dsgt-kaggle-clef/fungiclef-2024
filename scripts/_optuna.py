import optuna
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
import wandb

import random
# Load it to dataloader

torch.set_float32_matmul_precision('high')
N_CLASSES = 1717

train_df = pd.read_parquet("../train1.pq")
val_df = pd.read_parquet("../val1.pq")

# Load it as torch dataset
train_dataset = EmbeddingMetadataDataset(train_df)
valid_dataset = EmbeddingMetadataDataset(val_df)

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

# Class distribution for seesaw / CE loss
class_distribution = []
c = val_df.class_id.value_counts()
for i in range(N_CLASSES):
    class_distribution.append(c.get(i, 1))


BASE_BATCH_SIZE = 1024

EPOCHS = 15
WORKERS = 2

POISON_WEIGHT = 100

def objective(trial: optuna.trial.Trial):

    # BASE_LR = 1e-4
    
    UNKNOWN_LOSS_WEIGHT = 0 # trial.suggest_float("unknown_loss_weight", 0, 0.1, step=0.05)
    UNKNOWN_WEIGHT = 5 # trial.suggest_int("unknown_weight", 1, 3, step=1)
    POISON_LOSS_WEIGHT = trial.suggest_float("poison_loss_weight", 0, 0.2, step=0.05)
    SCHEDULER = "cosine" # trial.suggest_categorical("scheduler", ['cosine', 'reduce'])
    SEESAW_LOSS_Q = trial.suggest_float("seesaw_loss_q", 0.1, 0.5, step=0.1)

    BATCH_SIZE = 128 # 2 ** trial.suggest_int("batch_size", 6, 8)
    LR = 4e-5 # BASE_LR * BATCH_SIZE / BASE_BATCH_SIZE

    print("unknown: ", UNKNOWN_LOSS_WEIGHT, UNKNOWN_WEIGHT, "poison: ", POISON_LOSS_WEIGHT, SCHEDULER, "seesaw: ", SEESAW_LOSS_Q)

    model = FungiMEEModel(num_classes=N_CLASSES)

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

    run_name = "dino_optuna_" + datetime.now().strftime("%m%d%H%M")
    # Use our wrapper module to get a PyTorch Lightning trainer
    wandb_logger = WandbLogger(name=run_name, log_model=False, project="FungiClef")

    os.makedirs("../checkpoints/" + run_name, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="../checkpoints/" + run_name,
        save_top_k=5,
        monitor="valid_acc_cls",
        mode="max",
        save_weights_only=True,
    )

    trainer = L.Trainer(
        callbacks=[checkpoint_callback], logger=wandb_logger, max_epochs=EPOCHS
    )

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
    wandb.finish()

    return module.best_val_acc

study = optuna.create_study(
    directions=["maximize"],
    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
    study_name="dinov2-prod-4",
    load_if_exists=True
)
study.set_metric_names(["best_val_acc"])
study.optimize(objective, n_trials=1000, timeout=14400)
