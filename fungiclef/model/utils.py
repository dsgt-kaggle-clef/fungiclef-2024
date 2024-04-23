import timm
import torch.nn as nn

def get_timm_model(architecture_name, target_size, pretrained = False):
    """Helper function to get default model architectures using Timm
    """    
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg['classifier']
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))
    return net