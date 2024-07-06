import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from torchmetrics import Accuracy, Recall
from torchmetrics.classification import MulticlassF1Score
from fungiclef.utils import get_poison_mapping
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from torcheval.metrics.aggregation.auc import AUC
from fungiclef.model import NUM_CLASSES
from torch.nn import BCEWithLogitsLoss

N_CLASSES = NUM_CLASSES


class FungiDinoModule(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        learning_rate=0.001,
        loss=None,
        poison_weighting=100,
        unknown_weighting=5,
        poison_loss_weighting=1,
        unknown_loss_weighting=1,
        scheduler="cosine",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.lr = None
        self.scheduler = scheduler

        self.poison_mapping = torch.Tensor(get_poison_mapping()).cuda()

        self.loss = loss if loss else nn.functional.cross_entropy
        self.loss_unknown = BCEWithLogitsLoss(
            pos_weight=torch.Tensor([unknown_weighting])
        )
        self.loss_poison = BCEWithLogitsLoss(
            pos_weight=torch.Tensor([poison_weighting])
        )

        self.train_loss = []
        self.train_loss_cls = []
        self.train_loss_unknown = []
        self.train_loss_poison = []

        self.poison_loss_weighting = poison_loss_weighting
        self.unknown_loss_weighting = unknown_loss_weighting

        #  Metrics
        self.train_accuracy_class_top_1 = Accuracy(
            task="multiclass", num_classes=N_CLASSES
        )
        self.valid_accuracy_class_top_1 = Accuracy(
            task="multiclass", num_classes=N_CLASSES
        )

        self.train_f1 = MulticlassF1Score(num_classes=N_CLASSES, average="macro")
        self.valid_f1 = MulticlassF1Score(num_classes=N_CLASSES, average="macro")

        self.train_accuracy_class_top_3 = Accuracy(
            task="multiclass", num_classes=N_CLASSES, top_k=3
        )

        self.valid_accuracy_class_top_3 = Accuracy(
            task="multiclass", num_classes=N_CLASSES, top_k=3
        )

        self.train_poison_accuracy = Accuracy(task="binary")
        self.valid_poison_accuracy = Accuracy(task="binary")

        self.train_poison_recall = Recall(task="binary")
        self.valid_poison_recall = Recall(task="binary")

        self.train_unknown_auc = AUC()
        self.valid_unknown_auc = AUC()

        self.best_val_acc = 0

        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):

        (
            y_cls,
            y_pred_cls,
            y_pred_poison,
            y_pred_unknown,
            y_unknown,
            y_poison,
            loss,
            loss_cls,
            loss_unknown,
            loss_poison,
        ) = self.derive_loss(batch)

        self.train_loss.append(loss)
        self.train_loss_cls.append(loss_cls)
        self.train_loss_unknown.append(loss_unknown)
        self.train_loss_poison.append(loss_poison)

        y_pred_cls = y_pred_cls.softmax(dim=1)

        self.train_accuracy_class_top_1.update(y_pred_cls, y_cls)
        self.train_accuracy_class_top_3.update(y_pred_cls, y_cls)
        self.train_f1.update(y_pred_cls, y_cls)
        self.train_poison_accuracy.update(y_pred_poison, y_poison)
        self.train_poison_recall.update(y_pred_poison, y_poison)
        self.train_unknown_auc.update(y_pred_unknown, y_unknown)

        return loss

    def derive_loss(self, batch):
        emb, metadata, y_cls = batch
        y_pred = self.model.forward(emb, metadata)

        y_pred_cls = y_pred[:, :NUM_CLASSES]
        y_max = y_pred_cls.max(1)
        y_pred_poison_cls = self.poison_mapping[y_max.indices]
        y_pred_unknown_cls = (y_max.indices >= 1604).long()

        # y_pred_poison = y_pred[:, -2].squeeze()
        # y_pred_unknown = y_pred[:, -1].squeeze()

        y_unknown = metadata["unknown"].squeeze().long()
        y_poison = metadata["poisonous"].squeeze().long()
        y_cls = y_cls.squeeze()

        y_pred_poison = y_max.values * ((-1) ** (1 - y_pred_poison_cls))
        y_pred_unknown = y_max.values * ((-1) ** (1 - y_pred_unknown_cls))

        cls_loss = self.loss(y_pred_cls, y_cls)
        unknown_loss = self.loss_unknown(y_pred_unknown, y_unknown.float())
        poison_loss = self.loss_poison(y_pred_poison, y_poison.float())

        # print(y_pred_poison_cls, y_pred_poison, poison_loss)
        # print(y_pred_unknown_cls, y_pred_unknown, unknown_loss)

        # print(cls_loss.item(), unknown_loss.item(), poison_loss.item())

        loss = (
            cls_loss
            + self.poison_loss_weighting * poison_loss
            + self.unknown_loss_weighting * unknown_loss
        )
        y_pred_cls = y_pred_cls.softmax(dim=1)
        y_pred_poison = y_pred_poison.sigmoid()
        y_pred_unknown = y_pred_unknown.sigmoid()

        return (
            y_cls,
            y_pred_cls,
            y_pred_poison,
            y_pred_unknown,
            y_unknown,
            y_poison,
            loss,
            cls_loss,
            unknown_loss,
            poison_loss,
        )

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.train_loss).mean()
        avg_train_loss_cls = torch.stack(self.train_loss_cls).mean()
        avg_train_loss_unknown = torch.stack(self.train_loss_unknown).mean()
        avg_train_loss_poison = torch.stack(self.train_loss_poison).mean()

        self.log(
            "train_loss",
            avg_train_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_loss_cls",
            avg_train_loss_cls,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.log(
            "train_loss_unknown",
            avg_train_loss_unknown,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.log(
            "train_loss_poison",
            avg_train_loss_poison,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.log(
            "train_acc_cls",
            self.train_accuracy_class_top_1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_acc_cls_top_3",
            self.train_accuracy_class_top_3.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_acc_poison",
            self.train_poison_accuracy.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_recall_poison",
            self.train_poison_recall.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_f1",
            self.train_f1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_unknown_auc",
            self.train_unknown_auc.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.train_accuracy_class_top_1.reset()
        self.train_accuracy_class_top_3.reset()
        self.train_poison_accuracy.reset()
        self.train_f1.reset()
        self.train_poison_recall.reset()
        self.train_unknown_auc.reset()
        self.train_loss.clear()  # free memory
        self.train_loss_cls.clear()  # free memory
        self.train_loss_unknown.clear()  # free memory
        self.train_loss_poison.clear()  # free memory

    def validation_step(self, batch, batch_idx):

        (
            y_cls,
            y_pred_cls,
            y_pred_poison,
            y_pred_unknown,
            y_unknown,
            y_poison,
            loss,
            loss_cls,
            loss_unknown,
            loss_poison,
        ) = self.derive_loss(batch)

        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.log(
            "valid_loss_cls",
            loss_cls,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_loss_unknown",
            loss_unknown,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_loss_poison",
            loss_poison,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        valid_acc_cls = self.valid_accuracy_class_top_1(y_pred_cls, y_cls)
        valid_acc_cls_top_3 = self.valid_accuracy_class_top_3(y_pred_cls, y_cls)
        valid_f1 = self.valid_f1(y_pred_cls, y_cls)
        valid_acc_poison = self.valid_poison_accuracy(y_pred_poison, y_poison)
        valid_recall_poison = self.valid_poison_recall(y_pred_poison, y_poison)
        self.valid_unknown_auc.update(y_pred_unknown, y_unknown)
        valid_unknown_auc = self.valid_unknown_auc.compute()

        self.log(
            "valid_acc_cls",
            valid_acc_cls,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_acc_cls_top_3",
            valid_acc_cls_top_3,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_acc_poison",
            valid_acc_poison,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_recall_poison",
            valid_recall_poison,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_f1",
            valid_f1,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_unknown_auc",
            valid_unknown_auc,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.valid_accuracy_class_top_1.reset()
        self.valid_accuracy_class_top_3.reset()
        self.valid_f1.reset()
        self.valid_poison_accuracy.reset()

        if valid_acc_cls > self.best_val_acc:
            self.best_val_acc = valid_acc_cls

    def predict(self, batch, logits=False, device=0):

        img, metadata, y = batch
        y = y.long()
        x = x.to(device)
        y_pred = self.model.forward(img, metadata)

        y_pred_cls = y_pred[:, :N_CLASSES]

        if logits:
            return y_pred_cls
        else:
            return y_pred_cls.argmax(1).cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )

        if self.scheduler == "cosine":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingWarmRestarts(
                        optimizer, T_0=5, eta_min=5e-7
                    ),
                    "frequency": 1,
                },
            }
        elif self.scheduler == "reduce":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer),
                    "monitor": "valid_acc_cls",
                    "frequency": 1,
                },
            }


class FungiMetadataModule(L.LightningModule):
    # Simple wrapper for a torch nn.Module, with specified loss / optimizer.
    # This is supposed to be model agnostic.

    def __init__(self, model: nn.Module, optimizer=None, loss=None):
        super().__init__()
        self.model = model
        self.optimizer = (
            optimizer if optimizer else optim.Adam(self.parameters(), lr=1e-5)
        )

        self.poison_mapping = get_poison_mapping().to(self.device)

        self.loss = loss if loss else nn.functional.cross_entropy
        self.train_loss = []
        self.train_accuracy_class_top_1 = Accuracy(
            task="multiclass", num_classes=N_CLASSES
        )
        self.valid_accuracy_class_top_1 = Accuracy(
            task="multiclass", num_classes=N_CLASSES
        )
        self.train_f1 = MulticlassF1Score(num_classes=N_CLASSES, average="macro")
        self.valid_f1 = MulticlassF1Score(num_classes=N_CLASSES, average="macro")
        self.train_accuracy_class_top_3 = Accuracy(
            task="multiclass", num_classes=N_CLASSES, top_k=3
        )
        self.valid_accuracy_class_top_3 = Accuracy(
            task="multiclass", num_classes=N_CLASSES, top_k=3
        )
        self.train_accuracy_poison = Accuracy(task="binary")
        self.valid_accuracy_poison = Accuracy(task="binary")

        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):

        img, metadata, y_cls = batch
        y_pred = self.model.forward(img, metadata)
        loss = self.loss(y_pred, y_cls)
        self.train_loss.append(loss)

        y_pred_cls = y_pred.softmax(dim=1)

        assert (y_pred_cls >= 0).all(), "y_pred_cls contains negative values"
        assert (y_cls >= 0).all(), "y_cls contains negative values"

        poison_prediction = self.poison_mapping[
            y_pred_cls.argmax(1).to(self.poison_mapping.device)
        ]
        poison_gt = self.poison_mapping[y_cls.int().to(self.poison_mapping.device)]

        self.train_accuracy_class_top_1.update(y_pred_cls, y_cls)
        self.train_accuracy_class_top_3.update(y_pred_cls, y_cls)
        # print(y_pred_cls.shape, y_cls.shape)
        self.train_f1.update(y_pred_cls, y_cls)
        self.train_accuracy_poison.update(poison_prediction, poison_gt)

        return loss

    def on_train_epoch_end(self):
        all_train_loss = torch.stack(self.train_loss)
        # do something with all preds
        avg_train_loss = all_train_loss.mean()
        self.log(
            "train_loss",
            avg_train_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.log(
            "train_acc_cls",
            self.train_accuracy_class_top_1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_acc_cls_top_3",
            self.train_accuracy_class_top_3.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_acc_poison",
            self.train_accuracy_poison.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_f1",
            self.train_f1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.train_accuracy_class_top_1.reset()
        self.train_accuracy_class_top_3.reset()
        self.train_accuracy_poison.reset()
        self.train_f1.reset()
        self.train_loss.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        img, metadata, y_cls = batch
        y_cls = y_cls.long()
        y_pred = self.model.forward(img, metadata)
        loss = self.loss(y_pred, y_cls)

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True
        )

        y_pred_cls = y_pred.softmax(dim=1)

        poison_prediction = self.poison_mapping[
            y_pred_cls.argmax(1).to(self.poison_mapping.device)
        ]
        poison_gt = self.poison_mapping[y_cls.int().to(self.poison_mapping.device)]

        self.valid_accuracy_class_top_1.update(y_pred_cls, y_cls)
        self.valid_accuracy_class_top_3.update(y_pred_cls, y_cls)
        self.valid_f1.update(y_pred_cls, y_cls)
        self.valid_accuracy_poison.update(poison_prediction, poison_gt)

        self.valid_accuracy_class_top_1.compute()
        self.valid_accuracy_class_top_3.compute()
        self.valid_f1.compute()
        self.valid_accuracy_poison.compute()

        self.log(
            "valid_acc_cls",
            self.valid_accuracy_class_top_1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_acc_cls_top_3",
            self.valid_accuracy_class_top_3.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_acc_poison",
            self.valid_accuracy_poison.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_f1",
            self.valid_f1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.valid_accuracy_class_top_1.reset()
        self.valid_accuracy_class_top_3.reset()
        self.valid_f1.reset()
        self.valid_accuracy_poison.reset()

    def predict(self, batch, logits=False, device=0):

        img, metadata, y = batch
        y = y.long()
        x = x.to(device)
        y_pred = self.model.forward(img, metadata)

        y_pred_cls = y_pred[:, :N_CLASSES]

        if logits:
            return y_pred_cls
        else:
            return y_pred_cls.argmax(1).cpu().numpy()

    def configure_optimizers(self):
        optimizer = self.optimizer
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": ReduceLROnPlateau(optimizer),
            #     "monitor": "valid_acc_cls",
            #     "frequency": 1,
            #     # If "monitor" references validation metrics, then "frequency" should be set to a
            #     # multiple of "trainer.check_val_every_n_epoch".
            # },
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, eta_min=2e-6
                ),
                "frequency": 1,
                # "name": "cosine_lr"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


# define the LightningModule
class FungiModel(L.LightningModule):
    # Simple wrapper for a torch nn.Module, with specified loss / optimizer.
    # This is supposed to be model agnostic.

    def __init__(self, model: nn.Module, optimizer=None, loss=None, multi_output=False):
        super().__init__()
        self.model = model
        self.optimizer = (
            optimizer if optimizer else optim.Adam(self.parameters(), lr=1e-5)
        )

        self.poison_mapping = get_poison_mapping().to(self.device)

        self.loss = loss if loss else nn.functional.cross_entropy
        self.train_loss = []
        self.train_accuracy_class_top_1 = Accuracy(
            task="multiclass", num_classes=N_CLASSES
        )
        self.valid_accuracy_class_top_1 = Accuracy(
            task="multiclass", num_classes=N_CLASSES
        )
        self.train_f1 = MulticlassF1Score(num_classes=N_CLASSES, average="macro")
        self.valid_f1 = MulticlassF1Score(num_classes=N_CLASSES, average="macro")
        self.train_accuracy_class_top_3 = Accuracy(
            task="multiclass", num_classes=N_CLASSES, top_k=3
        )
        self.valid_accuracy_class_top_3 = Accuracy(
            task="multiclass", num_classes=N_CLASSES, top_k=3
        )
        self.train_accuracy_poison = Accuracy(task="binary")
        self.valid_accuracy_poison = Accuracy(task="binary")

        self.multi_output = multi_output

        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.train_loss.append(loss)

        y_pred_cls = y_pred[:, :N_CLASSES]
        y_cls = y

        y_pred_cls = y_pred_cls.softmax(dim=1)

        assert (y_pred_cls >= 0).all(), "y_pred_cls contains negative values"
        assert (y_cls >= 0).all(), "y_cls contains negative values"

        poison_prediction = self.poison_mapping[
            y_pred_cls.argmax(1).to(self.poison_mapping.device)
        ]
        poison_gt = self.poison_mapping[y_cls.int().to(self.poison_mapping.device)]

        self.train_accuracy_class_top_1.update(y_pred_cls, y_cls)
        self.train_accuracy_class_top_3.update(y_pred_cls, y_cls)
        # print(y_pred_cls.shape, y_cls.shape)
        self.train_f1.update(y_pred_cls, y_cls)
        self.train_accuracy_poison.update(poison_prediction, poison_gt)

        return loss

    def on_train_epoch_end(self):
        all_train_loss = torch.stack(self.train_loss)
        # do something with all preds
        avg_train_loss = all_train_loss.mean()
        self.log(
            "train_loss",
            avg_train_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.log(
            "train_acc_cls",
            self.train_accuracy_class_top_1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_acc_cls_top_3",
            self.train_accuracy_class_top_3.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_acc_poison",
            self.train_accuracy_poison.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "train_f1",
            self.train_f1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.train_accuracy_class_top_1.reset()
        self.train_accuracy_class_top_3.reset()
        self.train_accuracy_poison.reset()
        self.train_f1.reset()
        self.train_loss.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y = y.long()
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True
        )

        y_pred_cls = y_pred[:, :N_CLASSES]
        y_pred_cls = y_pred_cls.softmax(dim=1)

        y_cls = y

        poison_prediction = self.poison_mapping[
            y_pred_cls.argmax(1).to(self.poison_mapping.device)
        ]
        poison_gt = self.poison_mapping[y_cls.int().to(self.poison_mapping.device)]

        self.valid_accuracy_class_top_1.update(y_pred_cls, y_cls)
        self.valid_accuracy_class_top_3.update(y_pred_cls, y_cls)
        self.valid_f1.update(y_pred_cls, y_cls)
        self.valid_accuracy_poison.update(poison_prediction, poison_gt)

        self.valid_accuracy_class_top_1.compute()
        self.valid_accuracy_class_top_3.compute()
        self.valid_f1.compute()
        self.valid_accuracy_poison.compute()

        self.log(
            "valid_acc_cls",
            self.valid_accuracy_class_top_1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_acc_cls_top_3",
            self.valid_accuracy_class_top_3.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_acc_poison",
            self.valid_accuracy_poison.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            "valid_f1",
            self.valid_f1.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            on_epoch=True,
        )

        self.valid_accuracy_class_top_1.reset()
        self.valid_accuracy_class_top_3.reset()
        self.valid_f1.reset()
        self.valid_accuracy_poison.reset()

    def predict(self, batch, logits=False, device=0):
        x, y = batch
        y = y.long()
        x = x.to(device)
        y_pred = self.model(x)

        y_pred_cls = y_pred[:, :N_CLASSES]

        if logits:
            return y_pred_cls
        else:
            return y_pred_cls.argmax(1).cpu().numpy()

    def configure_optimizers(self):
        optimizer = self.optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "valid_acc_cls",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


# # init the autoencoder
# autoencoder = LitAutoEncoder(encoder, decoder)
