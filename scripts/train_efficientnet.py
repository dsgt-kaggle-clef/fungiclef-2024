from fungiclef._model._dataset import ImageDataset, EmbeddingDataset
from fungiclef._model.wrapper import FungiModel
from fungiclef._model.transforms import get_transforms
from fungiclef.utils import get_spark, spark_resource, read_config
import pandas as pd

from torch.utils.data import DataLoader
import lightning as L
import torch
from fungiclef._model.loss import FungiModelLoss

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
    transform=get_transforms(data="train"),
)
valid_dataset = ImageDataset(
    val_df, local_filepath="../../data/DF_300/", transform=get_transforms(data="valid")
)

# Define model. Here we use a simple stupid linear layer layer
from fungiclef._model.init_models import init_efficientnet_classifier

# N_CLASSES = len(
#     train_df.class_id.unique()
# )  # This should be 1605 - 1604 classes + 1 unknown class

N_CLASSES = 1605

model = init_efficientnet_classifier(n_classes=N_CLASSES)

_weights = model.state_dict()

# Load pretrained stuff
weights = torch.load("../checkpoints/efficientnet_b5_ce_base.pth")
weights["_fc.weight"] = _weights["_fc.weight"]
weights["_fc.bias"] = _weights["_fc.bias"]
model.load_state_dict(weights)

# Load it to dataloader
BATCH_SIZE = 48
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

# Use our wrapper module to get a PyTorch Lightning trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(log_model=False, project="FungiClef")

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints", save_top_k=2, monitor="val_loss"
)
trainer = L.Trainer(
    callbacks=[checkpoint_callback], logger=wandb_logger, max_epochs=EPOCHS
)

# Class distribution for seesaw loss
class_distribution = []
c = val_df.class_id.value_counts()
for i in range(1605):
    class_distribution.append(c.get(i, 0))

# Try use focal loss
loss = FungiModelLoss(loss="seesaw", class_distribution=class_distribution)
module = FungiModel(
    model,
    loss=loss.loss,
    optimizer=torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05),
)

trainer.fit(module, train_loader, valid_loader)
