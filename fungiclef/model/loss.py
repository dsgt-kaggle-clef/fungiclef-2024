from typing import Literal
from kornia.losses import focal_loss
from torch import nn
import torch
import torch.nn.functional as F


class FungiModelLoss:

    def __init__(self, loss: Literal["cross_entropy", "focal_loss"], class_weight=None):

        self.loss_fn = loss
        self.weight = class_weight

    def loss(self, pred, target):

        if self.loss_fn == "cross_entropy":
            if self.weight is not None:
                self.weight = self.weight.to(target.device)
            return F.cross_entropy(pred, target, weight=self.weight)
        elif self.loss_fn == "focal_loss":
            return focal_loss(pred, target, alpha=0.25, reduction="mean")

POISON_FALSE_NEGATIVE_WEIGHT = 100.0

class MultiTargetLoss(nn.Module):
    def __init__(self, idx_splits, output_weights=None, class_weights=None, device="cpu"):
        super().__init__()
        self.output_weights = (
            torch.Tensor(output_weights) if output_weights else torch.ones(len(idx_splits) + 1)
        ).to(device)
        self.idx_splits = idx_splits

        if class_weights is not None:
            _class_weights = []
            for class_weight in class_weights:
                if class_weight is not None:
                    _class_weights.append(torch.Tensor(class_weight).to(device))
                else:
                    _class_weights.append(None)

            self.class_weights = _class_weights
        else:
            self.class_weights = [None] * (len(idx_splits) + 1)


    def loss(self, pred, target):

        preds = torch.hsplit(pred, self.idx_splits)
        targets = torch.split(target, 1, dim=1)

        losses = []
        for i, (p, t) in enumerate(zip(preds, targets)):
            weight = self.class_weights[i].float() if self.class_weights[i] is not None else None
            
            t = t.squeeze(1).long()
            if(p.shape)[-1] == 1:
                p = p.squeeze(1)
                t = t.float()
                loss = F.binary_cross_entropy_with_logits(p, t, pos_weight=torch.tensor([POISON_FALSE_NEGATIVE_WEIGHT]))
            else:
                loss = F.cross_entropy(p, t, weight=weight)
            
            losses.append(loss * self.output_weights[i])

        return sum(losses)
