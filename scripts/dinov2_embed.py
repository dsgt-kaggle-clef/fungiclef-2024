from fungiclef.transforms import get_transforms
import pandas as pd

import torch

import timm

from PIL import Image
import os

import cv2

from tqdm import tqdm
from fungiclef.dataset import ImageMetadataDataset
import numpy as np
from torch.utils.data import DataLoader

train_df = pd.read_parquet("../train.pq")
val_df = pd.read_parquet("../val.pq")
_df = pd.concat((train_df, val_df))

DIM = 518
BASE_PATH = "../data/DF_FULL"

transforms = get_transforms(data="valid", width=DIM, height=DIM)

valid_dataset = ImageMetadataDataset(
    _df, local_filepath="../data/DF_FULL/", transform=transforms)

loader = DataLoader(valid_dataset, batch_size=3, shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = timm.create_model("timm/vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True)
model = model.to(device)
model.eval()

all_embs = []
for data in tqdm(loader):
    
    img, metadata, label = data
    img = img.to(device)

    emb = model.forward(img)

    all_embs.append(emb.detach().cpu().numpy())

all_embs = np.vstack(all_embs)

emb_df = pd.DataFrame(_df.image_path)

embs_list = [x for x in all_embs]
emb_df['embedding'] = embs_list
emb_df.to_parquet('dinov2_1024r.pq', index=False)
