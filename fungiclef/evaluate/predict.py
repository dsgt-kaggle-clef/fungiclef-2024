## Adapted from 

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

# custom script arguments
WIDTH, HEIGHT = 224, 224
IMG_COLUMN_NAME = "im_bytes"


class ImageDataset(Dataset):
    def __init__(self, df, model_mean, model_std, width, height):
        self.df = df
        self.transform = A.Compose(
            [
                A.Resize(width, height),
                A.Normalize(mean=model_mean, std=model_std),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # label = self.df["class_id"].values[idx]
        image = np.frombuffer(self.df[IMG_COLUMN_NAME][idx], dtype="uint8").reshape(
            (self.df["im_height"][idx], self.df["im_widgth"][idx], 3)
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        return image


@torch.no_grad()
def predict(model, testloader):
    """Iterate through test dataloader and run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    preds_all = []
    for imgs in tqdm(testloader, total=len(testloader)):
        imgs = imgs.to(device)
        preds = model(imgs)
        preds_all.append(preds.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    return preds_all



def run_inference_vit(input_df, output_path, model): 

    print("Creating DataLoader.")

    model_mean = list(model.default_cfg['mean'])
    model_std = list(model.default_cfg['std'])

    testset = ImageDataset(input_df, model_mean, model_std, WIDTH, HEIGHT)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    print("Running inference.")
    preds = predict(model, testloader)

    # save predictions
    print(f"Saving predictions to '{output_path}'.")
    user_pred_df = input_df[["observationID"]].copy()
    user_pred_df["class_id"] = preds.argmax(1)
    # (dummy example) convert instance-based prediction into observation-based predictions
    # by keeping predictions of first instances in the dataframe
    user_pred_df = user_pred_df.drop_duplicates("observationID", keep="first")
    user_pred_df.to_csv(output_path, index=False)

