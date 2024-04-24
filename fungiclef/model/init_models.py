import torch
import torch.nn as nn

from fungiclef.model.utils import get_timm_model
from efficientnet_pytorch import EfficientNet


# TODO: Make some crazy multi-head thing. This will do for now

VIT_MODEL_NAME = 'vit_large_patch16_224'
def get_vit_classifier(n_classes:int , pretrained_path=None, use_imagenet_pretrain=False):
    # Make simple classifier model with pretrained VIT
    pretrained_weights = None
    
    if pretrained_path: 
        pretrained_weights = torch.load(pretrained_path)
        use_imagenet_pretrain = False

    model = get_timm_model(VIT_MODEL_NAME, n_classes, pretrained=use_imagenet_pretrain)

    if pretrained_weights: 
        model.load_state_dict(pretrained_weights)

    return model

def get_efficientnet_classifier(n_classes:int , pretrained_path=None):
    # Make simple classifier model with EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b5')

    model._fc = nn.Linear(model._fc.in_features, n_classes)
    pretrained_weights = None

    if pretrained_weights: 
        model.load_state_dict(pretrained_weights)

    return model
    
    
    

def get_embedding_classifier(n_classes:int, embedding_size=384, checkpoint_path=None,):
    # Generates simple linear layer for classification
    
    model = nn.Linear(embedding_size, n_classes)
    
    if checkpoint_path: 
        weights = torch.load(checkpoint_path)
        model.load_state_dict(weights)

    return model


