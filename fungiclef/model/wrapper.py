import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from fungiclef.model.utils import get_poison_mapping

N_CLASSES = 1605


# define the LightningModule
class FungiModel(L.LightningModule):
    # Simple wrapper for a torch nn.Module, with specified loss / optimizer.
    # This is supposed to be model agnostic.

    def __init__(self, model: nn.Module, optimizer=None, loss=None, multi_output=False):
        super().__init__()
        self.model = model
        self.optimizer = (
            (
            optimizer if optimizer else optim.Adam(self.parameters(), lr=1e-5)
        )
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
        self.train_f1 = MulticlassF1Score(num_classes=N_CLASSES, average='macro')
        self.valid_f1 = MulticlassF1Score(num_classes=N_CLASSES, average='macro')
        self.train_accuracy_class_top_3 = Accuracy(
            
            task="multiclass", num_classes=N_CLASSES, top_k=3
        
        )
        self.valid_accuracy_class_top_3 = Accuracy(
            
            task="multiclass", num_classes=N_CLASSES, top_k=3
        
        )
        self.train_accuracy_poison = Accuracy(task="binary")
        self.valid_accuracy_poison = Accuracy(task="binary")

        self.multi_output = multi_output
        
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.train_loss.append(loss)

        y_pred_cls = y_pred[:, :N_CLASSES]
        y_cls = y[:, 0]

        poison_prediction = self.poison_mapping[
            y_pred_cls.argmax(1).to(self.poison_mapping.device)
        ]
        poison_gt = self.poison_mapping[y_cls.int().to(self.poison_mapping.device)]

        self.train_accuracy_class_top_1.update(y_pred_cls, y_cls)
        self.train_accuracy_class_top_3.update(y_pred_cls, y_cls)
        self.train_f1.update(y_pred_cls.argmax(1), y_cls)
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
        y_cls = y[:, 0]

        poison_prediction = self.poison_mapping[
            y_pred_cls.argmax(1).to(self.poison_mapping.device)
        ]
        poison_gt = self.poison_mapping[y_cls.int().to(self.poison_mapping.device)]

        self.valid_accuracy_class_top_1.update(y_pred_cls, y_cls)
        self.valid_accuracy_class_top_3.update(y_pred_cls, y_cls)
        self.valid_f1.update(y_pred_cls.argmax(1), y_cls)
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
        return optimizer


# class FungiMetadataModel(L.LightningModule):


# # init the autoencoder
# autoencoder = LitAutoEncoder(encoder, decoder)

