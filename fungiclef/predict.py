from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchdata.datapipes.iter import FileLister

import pyspark
from tqdm import tqdm

from utils import read_config, spark_resource

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
RANDOM_SEED = 0


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dinov2_vits14)
        self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data[0].transpose(1, 0).to(device)
        target = data[1].transpose(1, 0).to(device)

        if model.__class__.__name__ == 'FullTransformerTranslator':
            translation = model(source, target)
        else:
            translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)

def prepare_dataloader(src_path):
    if src_path is None:
        src_path = "gs://dsgt-clef-fungiclef-2024/data/parquet/DF20_embeddings/dino/data"

    with spark_resource as spark:

            # Step 1: Read the Parquet file
        df = spark.read.parquet(src_path)

        # Step 2: Convert to PyTorch tensor (assuming your target is to predict 'label')
        X = torch.tensor(df.drop('species', 'ImageUniqueID').values, dtype=torch.float32)
        y = torch.tensor(df['species'].values, dtype=torch.long)

        # Step 3: Create a custom Dataset
        class ParquetDataset(Dataset):
            def __init__(self, features, labels):
                self.features = features
                self.labels = labels

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]

        # Step 4: Instantiate the Dataset
        dataset = ParquetDataset(X, y)

        # Step 5: Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Example: Iterate over the DataLoader
        for batch_features, batch_labels in dataloader:
            # Process your batches here
            print(batch_features, batch_labels)


if __name__ == "__main__":
    config = read_config("config.json")
    src_path = config['gs_paths']['train']["embedded_dino"]
    model = DinoVisionTransformerClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataloader = prepare_dataloader(src_path=src_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    train(model, dataloader, optimizer, criterion, device=device)