import torch
import torch.nn as nn


def get_linear_classifier(n_classes:int, embedding_size=384, checkpoint_path=None,):
    # Generates simple linear layer for classification
    
    model = nn.Linear(embedding_size, n_classes)
    
    if checkpoint_path: 
        weights = torch.load(checkpoint_path)
        model.load_state_dict(weights)

    return model

