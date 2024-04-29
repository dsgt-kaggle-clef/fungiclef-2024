from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import io
import torch
import cv2

class ImageDataset(Dataset):
    def __init__(self, df, transform=None, local_filepath=None):
        self.df = df
        self.transform = transform
        self.local_filepath = local_filepath
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df['class_id'].values[idx]

        if self.local_filepath:
            file_path = self.local_filepath + self.df['image_path'].values[idx]

            image = cv2.imread(file_path)
            
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image = np.random.uniform(-1, 1, size=(299, 299, 3)).astype(np.float32)
                print(file_path)

        else:
        
            image = Image.open(io.BytesIO(self.df.data.values[idx]))
            image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class EmbeddingDataset(Dataset):
    def __init__(self, df, emb_key='embeddings'):
        self.df = df
        self.emb_key = emb_key
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df['class_id'].values[idx]
        
        embedding = np.array(self.df[self.emb_key].values[idx]).astype(np.float32)

        return embedding, label