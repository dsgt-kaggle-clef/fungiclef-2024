import torch
import torch.nn as nn

from fungiclef.model.utils import get_timm_model
from efficientnet_pytorch import EfficientNet

from fungiclef.model.cosine_classifier import CosineSimilarityLayer


# TODO: Make some crazy multi-head thing. This will do for now

VIT_MODEL_NAME = "vit_large_patch16_224"


def init_vit_classifier(
    n_classes: int, pretrained_path=None, use_imagenet_pretrain=False
):
    # Make simple classifier model with pretrained VIT
    pretrained_weights = None

    if pretrained_path:
        pretrained_weights = torch.load(pretrained_path)
        use_imagenet_pretrain = False

    model = get_timm_model(VIT_MODEL_NAME, n_classes, pretrained=use_imagenet_pretrain)

    if pretrained_weights:
        model.load_state_dict(pretrained_weights)

    return model


def init_efficientnet_classifier(n_classes: int, pretrained_path=None):
    # Make simple classifier model with EfficientNet
    model = EfficientNet.from_pretrained("efficientnet-b5")

    model._fc = nn.Linear(model._fc.in_features, n_classes)

    if pretrained_path:
        pretrained_weights = torch.load(pretrained_path)
        model.load_state_dict(pretrained_weights)

    return model


def init_embedding_classifier_linear(
    n_classes: int,
    embedding_size=384,
    checkpoint_path=None,
):
    # Generates simple linear layer for classification

    model = nn.Linear(embedding_size, n_classes)

    if checkpoint_path:
        weights = torch.load(checkpoint_path)
        model.load_state_dict(weights)

    return model


def init_embedding_classifier_mlp(
    n_classes: int,
    input_size=384,
    hidden_size=1024,
    checkpoint_path=None,
):
    # Generates a 2 layer MLP for classification
    model = nn.Sequential(
        nn.BatchNorm1d(input_size),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_classes),
    )

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.kaiming_normal_(param)

    if checkpoint_path:
        weights = torch.load(checkpoint_path)
        model.load_state_dict(weights)

    return model


def init_embedding_classifier_cosine(
    n_classes: int, embedding_size=384, checkpoint_path=None, embedding_path=None
):
    # Generates simple linear layer for classification

    model = CosineSimilarityLayer(
        embedding_size, n_classes, embedding_path=embedding_path
    )

    if checkpoint_path:
        weights = torch.load(checkpoint_path)
        model.load_state_dict(weights)

    return model
