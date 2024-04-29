from typing import Literal
from kornia.losses import focal_loss
from torch import nn

class FungiModelLoss():

    def __init__(self, loss: Literal['cross_entropy', 'focal_loss'], class_weight=None):

        self.loss_fn = loss
        self.weight = class_weight

    def loss(self, pred, target):

        if self.loss_fn == 'cross_entropy':
            if self.weight is not None:
                self.weight = self.weight.to(target.device)
            return nn.functional.cross_entropy(pred, target, weight=self.weight)
        elif self.loss_fn == "focal_loss":
            return focal_loss(pred, target, alpha=0.25, reduction="mean")
        
        