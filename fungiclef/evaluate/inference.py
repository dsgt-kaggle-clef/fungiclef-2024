import os
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn

UNKNOWN_CLASS = 1604

@torch.no_grad()
def generate_logits(model: nn.Module, test_loader):
    """Iterate through test dataloader and run inference. Outputs logits (in the event we want to do any postprocessing)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    preds_all = []
    for imgs, _ in tqdm(test_loader, total=len(test_loader)):
        imgs = imgs.to(device)
        preds = model(imgs)
        preds_all.append(preds.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    return preds_all

def predict_class(pred_logits: np.ndarray):
    """From logits, generate predictions. For now, we will only do simple straightforward argmax"""
    predicted_class = np.argmax(pred_logits, axis=1)
    predicted_class = np.where(predicted_class==UNKNOWN_CLASS, -1, predicted_class)

    return predicted_class