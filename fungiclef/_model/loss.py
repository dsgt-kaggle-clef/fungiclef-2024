from typing import Literal
from kornia.losses import focal_loss
from torch import nn
import torch
import torch.nn.functional as F


import numpy as np
import torch
import torch.nn as nn
from typing import Union

from torch import Tensor

class SeesawLoss:
    def __init__(self, class_distribution: Tensor,
                   num_classes: int,
                   p: float = 0.8,
                   q: float = 0.2,
                   eps: float = 1e-3,
                   reduction: str = 'mean',):
        
        """Calculate the Seesaw CrossEntropy loss.

        Args:
            cls_score (Tensor): The prediction with shape (N, C),
            C is the number of classes.
            labels (Tensor): The learning label of the prediction.
            label_weights (Tensor): Sample-wise loss weight.
            cum_samples (Tensor): Cumulative samples for each category.
            num_classes (int): The number of classes.
            p (float): The ``p`` in the mitigation factor.
            q (float): The ``q`` in the compenstation factor.
            eps (float): The minimal value of divisor to smooth
                the computation of compensation factor
            reduction (str, optional): The method used to reduce the loss.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            Tensor: The calculated loss
        """

        assert len(class_distribution) == num_classes
        
        self.class_distribution = class_distribution
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, cls_score: Tensor, labels: Tensor):

        cls_score = cls_score.to(self.class_distribution.device)
        labels = labels.to(self.class_distribution.device)

        assert cls_score.size(-1) == self.num_classes
        
        onehot_labels = F.one_hot(labels, self.num_classes)
        seesaw_weights = cls_score.new_ones(onehot_labels.size())

        # mitigation factor
        if self.p > 0:
            sample_ratio_matrix = self.class_distribution[None, :].clamp(
                min=1) / self.class_distribution[:, None].clamp(min=1)
            index = (sample_ratio_matrix < 1.0).float()
            sample_weights = sample_ratio_matrix.pow(self.p) * index + (1 - index)
            mitigation_factor = sample_weights[labels.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        # compensation factor
        if self.q > 0:
            scores = F.softmax(cls_score.detach(), dim=1)
            self_scores = scores[
                torch.arange(0, len(scores)).to(scores.device).long(),
                labels.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

        loss = F.cross_entropy(cls_score, labels, weight=None, reduction=self.reduction)

        return loss

        

class FungiModelLoss:

    def __init__(
        self,
        loss: Literal["cross_entropy", "focal_loss", "seesaw"],
        class_distribution=None,
    ):
        self.loss_fn = loss
        self.weight = None
        if class_distribution:
            _class_weights = 1 / torch.Tensor(np.array(class_distribution) / sum(class_distribution))
            self.weight = _class_weights / _class_weights.sum()

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
