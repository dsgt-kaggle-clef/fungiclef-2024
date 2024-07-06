import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import timm
from timm.layers import SelectAdaptivePool2d, LayerNorm2d
from timm.models.metaformer import MlpHead

DATE_SIZE = 4
GEO_SIZE = 7
SUBSTRATE_SIZE = 73
features = 768
NUM_CLASSES = 1717


class SquaredReLU(nn.Module):
    """
    Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(
        self,
        scale_value=1.0,
        bias_value=0.0,
        scale_learnable=True,
        bias_learnable=True,
        mode=None,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(
            scale_value * torch.ones(1), requires_grad=scale_learnable
        )
        self.bias = nn.Parameter(
            bias_value * torch.ones(1), requires_grad=bias_learnable
        )

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

# class FungiEmbeddingModel(nn.Module):
#     def __init__(
#         self,
#         metadata_weight=0.1,
#         num_classes=NUM_CLASSES,
#         dim=1024,
#     ):
#         super().__init__()

#         self.date_embedding = MlpHead(
#             dim=DATE_SIZE, num_classes=dim, mlp_ratio=128, act_layer=StarReLU
#         )
#         self.geo_embedding = MlpHead(
#             dim=GEO_SIZE, num_classes=dim, mlp_ratio=128, act_layer=StarReLU
#         )
#         self.substr_embedding = MlpHead(
#             dim=SUBSTRATE_SIZE,
#             num_classes=dim,
#             mlp_ratio=8,
#             act_layer=StarReLU,
#         )

#         self.date_weight = metadata_weight  # nn.Parameter(torch.tensor(0.1))
#         self.geo_weight = metadata_weight  # nn.Parameter(torch.tensor(0.1))
#         self.substr_weight = metadata_weight  # nn.Parameter(torch.tensor(0.1))

#         self.head = MlpHead(dim=dim, num_classes=num_classes, drop_rate=0)

#         for param in self.parameters():
#             if param.dim() > 1:
#                 nn.init.kaiming_normal_(param)


#     def forward(self, img_emb, metadata):
        
#         date_emb = self.date_embedding.forward(metadata["date"])
#         geo_emb = self.geo_embedding.forward(metadata["geo"])
#         substr_emb = self.substr_embedding.forward(metadata["substr"])

#         full_emb = (
#             date_emb * self.date_weight
#             + geo_emb * self.geo_weight
#             + substr_emb * self.substr_weight
#             + img_emb
#         )

#         return self.head.forward(full_emb)


class FungiMEEModel(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        dim=1024,
    ):
        super().__init__()

        print("Setting up Pytorch Model")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using devide: {self.device}")


        self.date_embedding = MlpHead(
            dim=DATE_SIZE, num_classes=dim, mlp_ratio=128, act_layer=StarReLU
        )
        self.geo_embedding = MlpHead(
            dim=GEO_SIZE, num_classes=dim, mlp_ratio=128, act_layer=StarReLU
        )
        self.substr_embedding = MlpHead(
            dim=SUBSTRATE_SIZE,
            num_classes=dim,
            mlp_ratio=8,
            act_layer=StarReLU,
        )

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True), num_layers=4)
        
        self.head = MlpHead(dim=dim, num_classes=num_classes, drop_rate=0)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)


    def forward(self, img_emb, metadata):

        img_emb = img_emb.to(self.device)
        
        date_emb = self.date_embedding.forward(metadata["date"].to(self.device))
        geo_emb = self.geo_embedding.forward(metadata["geo"].to(self.device))
        substr_emb = self.substr_embedding.forward(metadata["substr"].to(self.device))

        full_emb = torch.stack((img_emb, date_emb, geo_emb, substr_emb), dim=1) #.unsqueeze(0)
        # print(full_emb.shape)

        cls_emb = self.encoder.forward(full_emb)[:, 0, :].squeeze(1)

        return self.head.forward(cls_emb)
    
    def predict(self, img_emb, metadata):
        
        logits = self.forward(img_emb, metadata)

        # Any preprocess happens here

        return logits.argmax(1).tolist()
    
class FungiEnsembleModel(nn.Module):

    def __init__(self, models, softmax=True) -> None:
        super().__init__()

        self.models = nn.ModuleList()
        self.softmax = softmax
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for model in models:
            model = model.to(self.device)
            model.eval()
            self.models.append(model)
        
    def forward(self, img_emb, metadata):

        img_emb = img_emb.to(self.device)

        probs = []        

        for model in self.models:
            logits = model.forward(img_emb, metadata)
            
            p = logits.softmax(dim=1).detach().cpu() if self.softmax else logits.detach().cpu()

            probs.append(p)

        return torch.stack(probs).mean(dim=0)
    
    def predict(self, img_emb, metadata):
        
        logits = self.forward(img_emb, metadata)

        # Any preprocess happens here

        return logits.argmax(1).tolist()
    

# class FungiMetadataModel(nn.Module):
#     def __init__(
#         self,
#         model_name="caformer_b36.sail_in22k",
#         metadata_weight=0.1,
#         use_metadata=True,
#     ):
#         super().__init__()

#         self.use_metadata = use_metadata

#         if use_metadata:
#             self.date_embedding = nn.Sequential(
#                 nn.Linear(DATE_SIZE, features), nn.GELU()
#             )
#             self.geo_embedding = nn.Sequential(
#                 nn.Linear(GEO_SIZE, features), nn.GELU()
#             )
#             self.substr_embedding = nn.Sequential(
#                 nn.Linear(SUBSTRATE_SIZE, features), nn.GELU()
#             )

#             self.date_embedding = MlpHead(
#                 dim=DATE_SIZE, num_classes=features, mlp_ratio=4, act_layer=StarReLU
#             )
#             self.geo_embedding = MlpHead(
#                 dim=GEO_SIZE, num_classes=features, mlp_ratio=4, act_layer=StarReLU
#             )
#             self.substr_embedding = MlpHead(
#                 dim=SUBSTRATE_SIZE,
#                 num_classes=features,
#                 mlp_ratio=4,
#                 act_layer=StarReLU,
#             )

#             self.date_weight = metadata_weight  # nn.Parameter(torch.tensor(0.1))
#             self.geo_weight = metadata_weight  # nn.Parameter(torch.tensor(0.1))
#             self.substr_weight = metadata_weight  # nn.Parameter(torch.tensor(0.1))

#             self.head = MlpHead(dim=features, num_classes=NUM_CLASSES, drop_rate=0)

#             for param in self.parameters():
#                 if param.dim() > 1:
#                     nn.init.kaiming_normal_(param)

#             self.img_model = timm.create_model(
#                 model_name, pretrained=True, num_classes=0
#             )

#         else:

#             self.img_model: nn.Module = timm.create_model(
#                 model_name, pretrained=True, num_classes=NUM_CLASSES
#             )

#     def forward(self, img, metadata):

#         if self.use_metadata:

#             date_emb = self.date_embedding.forward(metadata["date"])
#             geo_emb = self.geo_embedding.forward(metadata["geo"])
#             substr_emb = self.substr_embedding.forward(metadata["substr"])

#             img_emb = self.img_model.forward(img)

#             full_emb = img_emb
#             full_emb = (
#                 date_emb * self.date_weight
#                 + geo_emb * self.geo_weight
#                 + substr_emb * self.substr_weight
#                 + img_emb
#             )

#             return self.head.forward(full_emb)
        
#         else:
        
#             return self.img_model.forward(img)
