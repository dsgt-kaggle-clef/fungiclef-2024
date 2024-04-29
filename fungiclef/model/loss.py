from typing import Literal
from kornia.losses import focal_loss
from torch import nn

import numpy as np
import torch
import torch.nn as nn
from typing import Union


## From https://github.com/bamps53/SeesawLoss/blob/master/seesaw_loss.py


class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.

    Args:
    class_counts: The list which has number of samples for each class.
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: Union[list, np.array], p: float = 0.8):
        super().__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)

        self.num_classes = len(class_counts)

        self.eps = 1.0e-6

    def forward(self, logits, targets):
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)

        if len(targets.shape) == 1:
            _targets = (
                torch.zeros(targets.shape[0], self.num_classes).int().to(targets.device)
            )
            _targets[:, targets] = 1
            targets = _targets

        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]
        ).sum(axis=-1) + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (-targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class DistibutionAgnosticSeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.

    Args:
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 for default following the paper.
    """

    def __init__(self, p: float = 0.8):
        super().__init__()
        self.eps = 1.0e-6
        self.p = p
        self.s = None
        self.class_counts = None

    def forward(self, logits, targets):
        if self.class_counts is None:
            self.class_counts = targets.sum(axis=0) + 1  # to prevent devided by zero.
        else:
            self.class_counts += targets.sum(axis=0)

        conditions = self.class_counts[:, None] > self.class_counts[None, :]
        trues = (self.class_counts[None, :] / self.class_counts[:, None]) ** self.p
        falses = torch.ones(len(self.class_counts), len(self.class_counts))
        self.s = torch.where(conditions, trues, falses)

        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]
        ).sum(axis=-1) + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (-targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class FungiModelLoss:

    def __init__(
        self,
        loss: Literal["cross_entropy", "focal_loss", "seesaw"],
        class_distribution=None,
    ):

        if loss == "seesaw":
            if class_distribution:
                self.seesaw = SeesawLossWithLogits(class_distribution)
            else:
                self.seesaw = DistibutionAgnosticSeesawLossWithLogits()

        self.loss_fn = loss
        self.weight = class_weight

    def loss(self, pred, target):
        if self.loss_fn == "cross_entropy":
            return nn.functional.cross_entropy(pred, target)
        elif self.loss_fn == "focal_loss":
            return focal_loss(pred, target, alpha=0.25, reduction="mean")
        elif self.loss_fn == "seesaw":
            return self.seesaw(pred, target)
