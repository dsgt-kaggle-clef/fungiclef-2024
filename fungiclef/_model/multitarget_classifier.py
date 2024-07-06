import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

class FungiMetadataModel(nn.Module):
    def __init__(
        self, input_size, num_classes_list, hidden_size=1024
    ):
        super().__init__()

        self.models = nn.ModuleList()

        for num_classes in num_classes_list:
            model = nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, num_classes),
                )
            
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.kaiming_normal_(param)
            self.models.append(model)
        
        
    def forward(self, x):
        
        outputs = []

        for model in self.models:
            outputs.append(model.forward(x))

        return torch.hstack(outputs)


class MultiTargetClassifier(nn.Module):
    def __init__(
        self, input_size, num_classes_list, hidden_size=1024
    ):
        super().__init__()

        self.models = nn.ModuleList()

        for num_classes in num_classes_list:
            model = nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, num_classes),
                )
            
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.kaiming_normal_(param)
            self.models.append(model)
        
        
    def forward(self, x):
        
        outputs = []

        for model in self.models:
            outputs.append(model.forward(x))

        return torch.hstack(outputs)
