from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import io
import torch
import cv2
import os 
import pandas as pd

TIME = ['m0', 'm1', 'd0', 'd1']
GEO = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g_float']
SUBSTRATE = ["substrate_0",
    "substrate_1",
    "substrate_2",
    "substrate_3",
    "substrate_4",
    "substrate_5",
    "substrate_6",
    "substrate_7",
    "substrate_8",
    "substrate_9",
    "substrate_10",
    "substrate_11",
    "substrate_12",
    "substrate_13",
    "substrate_14",
    "substrate_15",
    "substrate_16",
    "substrate_17",
    "substrate_18",
    "substrate_19",
    "substrate_20",
    "substrate_21",
    "substrate_22",
    "substrate_23",
    "substrate_24",
    "substrate_25",
    "substrate_26",
    "substrate_27",
    "substrate_28",
    "substrate_29",
    "substrate_30",
    "metasubstrate_0",
    "metasubstrate_1",
    "metasubstrate_2",
    "metasubstrate_3",
    "metasubstrate_4",
    "metasubstrate_5",
    "metasubstrate_6",
    "metasubstrate_7",
    "metasubstrate_8",
    "metasubstrate_9",
    "habitat_0",
    "habitat_1",
    "habitat_2",
    "habitat_3",
    "habitat_4",
    "habitat_5",
    "habitat_6",
    "habitat_7",
    "habitat_8",
    "habitat_9",
    "habitat_10",
    "habitat_11",
    "habitat_12",
    "habitat_13",
    "habitat_14",
    "habitat_15",
    "habitat_16",
    "habitat_17",
    "habitat_18",
    "habitat_19",
    "habitat_20",
    "habitat_21",
    "habitat_22",
    "habitat_23",
    "habitat_24",
    "habitat_25",
    "habitat_26",
    "habitat_27",
    "habitat_28",
    "habitat_29",
    "habitat_30",
    "habitat_31",
]

class ImageMetadataDataset(Dataset):
    def __init__(self, df, transform=None, local_filepath=None):
        self.df = df
        self.transform = transform
        self.local_filepath = local_filepath
        
        self.filepaths = df["image_path"].apply(lambda x: x.replace("jpg", "JPG")).to_list()
        self.metadata_date = df[TIME].to_numpy()
        self.metadata_geo = df[GEO].to_numpy()
        self.metadata_substrate = df[SUBSTRATE].to_numpy()
        self.poisonous = df['poisonous'].to_numpy()
        self.unknown = df['unknown'].to_numpy()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df["class_id"].values[idx]

        file_path = os.path.join(self.local_filepath, self.filepaths[idx])

        try:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(file_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        metadata = {
            "date": torch.from_numpy(self.metadata_date[idx, :]).type(torch.float),
            "geo": torch.from_numpy(self.metadata_geo[idx, :]).type(torch.float),
            "substr": torch.from_numpy(self.metadata_substrate[idx, :]).type(torch.float),
            'poisonous': torch.Tensor([self.poisonous[idx]]).long(),
            'unknown': torch.Tensor([self.unknown[idx]]).long(),
        }

        return image, metadata, label

class EmbeddingMetadataDataset(Dataset):
    def __init__(self, df):
        self.df = df

        self.emb = df['embedding']
        self.metadata_date = df[TIME].to_numpy()
        self.metadata_geo = df[GEO].to_numpy()
        self.metadata_substrate = df[SUBSTRATE].to_numpy()
        self.poisonous = df['poisonous'].to_numpy()
        self.unknown = df['unknown'].to_numpy()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.Tensor([self.df["class_id"].values[idx]]).long()

        embedding = torch.Tensor(self.emb[idx].copy()).type(torch.float)

        metadata = {
            "date": torch.from_numpy(self.metadata_date[idx, :]).type(torch.float),
            "geo": torch.from_numpy(self.metadata_geo[idx, :]).type(torch.float),
            "substr": torch.from_numpy(self.metadata_substrate[idx, :]).type(torch.float),
            'poisonous': torch.Tensor([self.poisonous[idx]]).long(),
            'unknown': torch.Tensor([self.unknown[idx]]).long(),
        }

        return embedding, metadata, label