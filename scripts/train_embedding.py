from fungiclef.model.dataset import ImageDataset, EmbeddingDataset
from fungiclef.model.wrapper import FungiModel
from fungiclef.model.transforms import get_transforms
from fungiclef.utils import get_spark, spark_resource, read_config
import pandas as pd

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader
import lightning as L
import torch
from fungiclef.model.loss import FungiModelLoss
from fungiclef.model.init_models import init_embedding_classifier_linear, init_embedding_classifier_cosine, init_embedding_classifier_mlp

from torch import optim, nn, utils, Tensor

RESNET_TRAIN = "../production_resnet_DF_300_train.parquet"
RESNET_VAL = "../production_resnet_DF_300_valid.parquet"

# train_df = pd.read_parquet(RESNET_TRAIN)
# val_df = pd.read_parquet(RESNET_VAL)

# train_dataset = EmbeddingDataset(train_df, emb_key="embeddings")
# valid_dataset = EmbeddingDataset(val_df, emb_key="embeddings")

DINO_TRAIN = "../production_dino_cls_DF_300_train.parquet"
DINO_VAL = "../production_dino_cls_DF_300_valid.parquet"

train_df = pd.read_parquet(DINO_TRAIN)
val_df = pd.read_parquet(DINO_VAL)

# Load it as torch dataset
train_dataset = EmbeddingDataset(train_df, emb_key="embedding")
valid_dataset = EmbeddingDataset(val_df, emb_key="embedding")

N_CLASSES = len(
    train_df.class_id.unique()
)  # This should be 1605 - 1604 classes + 1 unknown class

N_CLASSES = 1605

# model = init_embedding_classifier_cosine(n_classes=N_CLASSES, embedding_size=768) #, embedding_path="../production_dino_cls_DF_300_train.parquet")
model = init_embedding_classifier_mlp(n_classes=N_CLASSES, input_size=768)

# Load it to dataloader
BATCH_SIZE = 512
# Adjust BATCH_SIZE and ACCUMULATION_STEPS to values that if multiplied results in 64
EPOCHS = 100
WORKERS = 4

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
)
# Use our wrapper module to get a PyTorch Lightning trainer

module = FungiModel(model)
trainer = L.Trainer(accelerator="gpu", max_epochs=1)

wandb_logger = WandbLogger(log_model=False, project="FungiClef")

# checkpoint_callback = ModelCheckpoint(
#     dirpath="../checkpoints", save_top_k=2, monitor="val_loss"
# )
# trainer = L.Trainer(
#     callbacks=[checkpoint_callback], logger=wandb_logger, max_epochs=100
# )

trainer = L.Trainer(
    logger=wandb_logger, max_epochs=100
)


# Use class weight?
c = val_df.class_id.value_counts()
class_weight = []

for i in range(1605):
    class_weight.append(c.get(i, 0) / c.sum())

class_weight = torch.Tensor(class_weight)

# Try use focal loss
loss = FungiModelLoss(loss="cross_entropy", class_weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
module = FungiModel(model, loss=loss.loss, optimizer=optimizer)

trainer.fit(module, train_loader, valid_loader)