import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

class CosineSimilarityLayer(nn.Module):
    def __init__(self, num_features, num_classes, embedding_path=None):
        super().__init__()
        # Create a parameter matrix for class embeddings

        self.num_classes = num_classes

        if embedding_path:
            self.class_embeddings = nn.Parameter(self.compute_mean_embeddings(embedding_path))
        else:
            self.class_embeddings = nn.Parameter(torch.randn(num_classes, num_features))
            nn.init.xavier_uniform_(self.class_embeddings)

    def compute_mean_embeddings(self, embedding_path):

        print("Computing embeddings")

        df = pd.read_parquet(embedding_path)

        mean_embeddings = []

        for i in tqdm(range(self.num_classes)):
            mean_embeddings.append(
                torch.Tensor(list(df[df.class_id == i].embedding)).mean(axis=0)
            )

        return torch.stack(mean_embeddings)

    def forward(self, x):
        # Normalize the input features and the class embeddings to unit vectors along the feature dimension
        x_normalized = F.normalize(x, p=2, dim=1)
        class_embeddings_normalized = F.normalize(self.class_embeddings, p=2, dim=1)
        
        # Compute cosine similarity between the input features and class embeddings
        cosine_similarity = torch.mm(x_normalized, class_embeddings_normalized.t())
        
        return cosine_similarity