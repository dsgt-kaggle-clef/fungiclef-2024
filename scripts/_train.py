from fungiclef._model._dataset import ImageDataset, EmbeddingDataset
from fungiclef._model.wrapper import FungiModel
from fungiclef._model.transforms import get_transforms
from fungiclef._model.utils import get_timm_model
from fungiclef.utils import get_spark, spark_resource, read_config
import pandas as pd

from torch.utils.data import DataLoader
import lightning as L
import torch
from fungiclef._model.loss import FungiModelLoss
import timm

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import v2
from torch.utils.data import default_collate
import random

# Load it to dataloader
BATCH_SIZE = 24
EPOCHS = 50
WORKERS = 8
N_CLASSES = 1605
LR = 2e-5

CUTMIX_MIXUP_P = 0.25

train_df = pd.read_csv("../metadata_train.csv")
val_df = pd.read_csv("../metadata_val.csv")
test_df = pd.read_csv("../metadata_test.csv")

# train_df = pd.read_csv("../metadata_train_test.csv")
# val_df = pd.read_csv("../metadata_val_test.csv")
# test_df = pd.read_csv("../metadata_test.csv")

# Load it as torch dataset
train_dataset = ImageDataset(
    train_df,
    local_filepath="../data/DF_FULL/",
    transform=get_transforms(data="train"),
)
valid_dataset = ImageDataset(
    val_df, local_filepath="../data/DF_FULL/", transform=get_transforms(data="valid")
)

# cutmix = v2.CutMix(num_classes=N_CLASSES)
# mixup = v2.MixUp(num_classes=N_CLASSES)
# cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
# identity = v2.Identity()
erase = v2.RandomErasing(p=CUTMIX_MIXUP_P)


def collate_fn(batch):
    return erase(*default_collate(batch))
    return cutmix_or_mixup(
        *default_collate(batch)
    )  # if random.random() < CUTMIX_MIXUP_P else identity(*default_collate(batch))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    collate_fn=collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
)

model = timm.create_model(
    "caformer_b36.sail_in22k", pretrained=True, num_classes=N_CLASSES
)

# Use our wrapper module to get a PyTorch Lightning trainer
wandb_logger = WandbLogger(log_model=False, project="FungiClef")

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints", save_top_k=3, monitor="valid_acc_cls", mode="max"
)
trainer = L.Trainer(
    callbacks=[checkpoint_callback], logger=wandb_logger, max_epochs=EPOCHS
)

# Class distribution for seesaw loss
class_distribution = []
c = val_df.class_id.value_counts()
for i in range(N_CLASSES):
    class_distribution.append(c.get(i, 0))

# Try use focal loss
loss = FungiModelLoss(loss="cross_entropy", class_distribution=class_distribution)
module = FungiModel(
    model,
    loss=loss.loss,
    optimizer=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05),
)

trainer.fit(module, train_loader, valid_loader)
