import torch

from fungiclef.model.utils import get_timm_model

MODEL_NAME = 'vit_large_patch16_224'

# TODO: Make some crazy multi-head thing. This will do for now

def get_simple_classifier(n_classes:int , pretrained_path=None, use_imagenet_pretrain=False):
    # Make simple classifier model 
    pretrained_weights = None
    
    if pretrained_path: 
        pretrained_weights = torch.load(pretrained_path)
        use_imagenet_pretrain = False

    model = get_timm_model(MODEL_NAME, n_classes, pretrained=use_imagenet_pretrain)

    if pretrained_weights: 
        model.load_state_dict(pretrained_weights)

    return model

