import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L


# define the LightningModule
class FungiModel(L.LightningModule):
    # Simple wrapper for a torch nn.Module, with specified loss / optimizer.
    # This is supposed to be model agnostic. 

    def __init__(self, model: nn.Module, optimizer=None, loss=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer if optimizer else optim.Adam(self.parameters(), lr=1e-3)
        self.loss = loss if loss else nn.functional.cross_entropy
        
    def training_step(self, batch, batch_idx):
        
        
        x, y = batch
        y = y.long()
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y = y.long()
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# # init the autoencoder
# autoencoder = LitAutoEncoder(encoder, decoder)