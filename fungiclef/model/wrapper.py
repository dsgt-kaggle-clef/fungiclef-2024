import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from torchmetrics import Accuracy
from fungiclef.model.utils import get_poison_mapping

N_CLASSES = 1605


# define the LightningModule
class FungiModel(L.LightningModule):
    # Simple wrapper for a torch nn.Module, with specified loss / optimizer.
    # This is supposed to be model agnostic. 

    def __init__(self, model: nn.Module, optimizer=None, loss=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer if optimizer else optim.Adam(self.parameters(), lr=1e-3)

        self.poison_mapping = get_poison_mapping().to(self.device)

        self.loss = loss if loss else nn.functional.cross_entropy
        self.train_loss = []
        self.train_accuracy_class_top_1 = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.valid_accuracy_class_top_1 = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.train_accuracy_class_top_3 = Accuracy(task="multiclass", num_classes=N_CLASSES, top_k=3)
        self.valid_accuracy_class_top_3 = Accuracy(task="multiclass", num_classes=N_CLASSES, top_k=3)
        self.train_accuracy_poison = Accuracy(task="binary")
        self.valid_accuracy_poison =  Accuracy(task="binary")
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.train_loss.append(loss)
        
        poison_prediction = self.poison_mapping[y_pred.argmax(1).to(self.poison_mapping.device)]
        poison_gt = self.poison_mapping[y.int().to(self.poison_mapping.device)]
        
        self.train_accuracy_class_top_1.update(y_pred, y)
        self.train_accuracy_class_top_3.update(y_pred, y)
        self.train_accuracy_poison.update(poison_prediction, poison_gt)

        return loss
    
    def on_train_epoch_end(self):
        all_train_loss = torch.stack(self.train_loss)
        # do something with all preds
        print(all_train_loss.shape)
        avg_train_loss = all_train_loss.mean()
        self.log("train_loss", avg_train_loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        
        self.train_accuracy_class_top_1.compute()
        self.train_accuracy_class_top_3.compute()
        self.train_accuracy_poison.compute()

        self.log("train_acc_cls", self.train_accuracy_class_top_1.compute(), prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("train_acc_cls_top_3", self.train_accuracy_class_top_3.compute(), prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("train_acc_poison", self.train_accuracy_poison.compute(), prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
                
        self.train_accuracy_class_top_1.reset()
        self.train_accuracy_class_top_3.reset()
        self.train_accuracy_poison.reset()
        self.train_loss.clear()  # free memory
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y = y.long()
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)

        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

        poison_prediction = self.poison_mapping[y_pred.argmax(1).to(self.poison_mapping.device)]
        poison_gt = self.poison_mapping[y.int().to(self.poison_mapping.device)]
        
        self.valid_accuracy_class_top_1.update(y_pred, y)
        self.valid_accuracy_class_top_3.update(y_pred, y)
        self.valid_accuracy_poison.update(poison_prediction, poison_gt)

        self.valid_accuracy_class_top_1.compute()
        self.valid_accuracy_class_top_3.compute()
        self.valid_accuracy_poison.compute()

        self.log("valid_acc_cls", self.valid_accuracy_class_top_1.compute(), prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("valid_acc_cls_top_3", self.valid_accuracy_class_top_3.compute(), prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("valid_acc_poison", self.valid_accuracy_poison.compute(), prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
                
        self.valid_accuracy_class_top_1.reset()
        self.valid_accuracy_class_top_3.reset()
        self.valid_accuracy_poison.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# # init the autoencoder
# autoencoder = LitAutoEncoder(encoder, decoder)