from fungiclef._model._dataset import ImageDataset, EmbeddingDataset
from fungiclef._model.wrapper import FungiModel
from fungiclef._model.transforms import get_transforms
from fungiclef.utils import get_spark, spark_resource, read_config
import pandas as pd
from fungiclef._model.init_models import (
    init_efficientnet_classifier,
    init_vit_classifier,
)

from torch.utils.data import DataLoader
import lightning as L
import torch
from fungiclef._model.loss import FungiModelLoss

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

train_df = pd.read_csv("../metadata_train.csv")
val_df = pd.read_csv("../metadata_val.csv")
test_df = pd.read_csv("../metadata_test.csv")

# train_df = pd.read_csv("../trial_metadata.csv")
# val_df = pd.read_csv("../trial_metadata.csv")
# test_df = pd.read_csv("../trial_metadata.csv")

# Load it as torch dataset
train_dataset = ImageDataset(
    train_df,
    local_filepath="../../data/DF_300/",
    transform=get_transforms(data="train", width=224, height=224),
)
valid_dataset = ImageDataset(
    val_df,
    local_filepath="../../data/DF_300/",
    transform=get_transforms(data="valid", width=224, height=224),
)


# N_CLASSES = len(
#     train_df.class_id.unique()
# )  # This should be 1605 - 1604 classes + 1 unknown class

N_CLASSES = 1605
PRETRAINED_PATH = "../checkpoints/vit_16L_224_base.pth"

model = init_vit_classifier(n_classes=N_CLASSES, pretrained_path=PRETRAINED_PATH)

# Load it to dataloader
BATCH_SIZE = 64
EPOCHS = 20
WORKERS = 3


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
)

wandb_logger = WandbLogger(log_model=False, project="FungiClef")

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints", save_top_k=2, monitor="val_loss"
)
trainer = L.Trainer(
    callbacks=[checkpoint_callback], logger=wandb_logger, max_epochs=EPOCHS
)

# Try use focal loss
loss = FungiModelLoss(loss="focal_loss")
module = FungiModel(model, loss=loss.loss)

trainer.fit(module, train_loader, valid_loader)
