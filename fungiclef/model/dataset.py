from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import io
import torch

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df['class_id'].values[idx]
        
        image = Image.open(io.BytesIO(self.df.data.values[idx]))
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df['class_id'].values[idx]
        
        embedding = np.array(self.df['embeddings'].values[idx])

        return embedding, label