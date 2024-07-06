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

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
import tqdm

from fungiclef.logger import init_logger
import wandb

WANDB_PROJECT_NAME = "FungiClef"

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_PARQUET = "dev_train.parquet"
VAL_PARQUET = "dev_val.parquet"

EPOCHS = 20
BATCH_SIZE = 32
ACCUMULATION_STEPS = 64 // BATCH_SIZE
WORKERS = 4

LR = 0.01

train_df = pd.read_parquet(TRAIN_PARQUET)
val_df = pd.read_parquet(VAL_PARQUET)
logger = init_logger()


wandb.init(
    project=WANDB_PROJECT_NAME, 
    name="vit_test", # TODO: Config name here as well
    config=None, # TODO: To add config here
)


def train_model(model, train_dataset, valid_dataset):    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # TODO: Move these outside tto config files etc.

    model.to(device)
    
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_score = 0.
    best_loss = np.inf

    for epoch in range(EPOCHS):

        start_time = time.time()

        model.train()
        avg_loss = 0.

        optimizer.zero_grad()

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):

            images = images.to(device)
            labels = labels.to(device)

            y_preds = model(images)

            loss = criterion(y_preds, labels)

            # Scale the loss to the mean of the accumulated batch size
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            if (i - 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                avg_loss += loss.item() / len(train_loader)

        model.eval()
        avg_val_loss = 0.
        preds = np.zeros((len(valid_dataset)))
        preds_raw = []

        for i, (images, labels) in enumerate(valid_loader):
            
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                y_preds = model(images)
            
            preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()
            preds_raw.extend(y_preds.to('cpu').numpy())

            loss = criterion(y_preds, labels)
            avg_val_loss += loss.item() / len(valid_loader)
        
        scheduler.step(avg_val_loss)
            
        score = f1_score(val_df['class_id'], preds, average='macro')
        accuracy = accuracy_score(val_df['class_id'], preds)
        recall_3 = top_k_accuracy_score(val_df['class_id'], preds_raw, k=3)

        elapsed = time.time() - start_time

        logger.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} F1: {score:.6f}  Accuracy: {accuracy:.6f} Recall@3: {recall_3:.6f} time: {elapsed:.0f}s')

        if accuracy>best_score:
            best_score = accuracy
            logger.debug(f'  Epoch {epoch+1} - Save Best Accuracy: {best_score:.6f} Model')
            torch.save(model.state_dict(), f'checkpoints/DF20-ViT_large_patch16_384_best_accuracy.pth')

        if avg_val_loss<best_loss:
            best_loss = avg_val_loss
            logger.debug(f'  Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(model.state_dict(), f'checkpoints/DF20-ViT_large_patch16_384_best_loss.pth')


N_CLASSES = len(train_df['class_id'].unique())
