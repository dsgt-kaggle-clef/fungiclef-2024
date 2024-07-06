from fungiclef._model._dataset import (
    ImageDataset,
    EmbeddingDataset,
    EmbeddingMetadataDataset,
)
from fungiclef._model.wrapper import FungiModel
from fungiclef._model.transforms import get_transforms
from fungiclef._model.multitarget_classifier import MultiTargetClassifier
from fungiclef.utils import get_spark, spark_resource, read_config
import pandas as pd

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L
import torch
from fungiclef._model.loss import FungiModelLoss, MultiTargetLoss
from fungiclef._model.init_models import (
    init_embedding_classifier_linear,
    init_embedding_classifier_cosine,
    init_embedding_classifier_mlp,
)

from torch import optim, nn, utils, Tensor
import numpy as np
import wandb

DINO_TRAIN = "../DF_300_metadata_train.parquet"
DINO_VAL = "../DF_300_metadata_valid.parquet"

EPOCHS = 20
BATCH_SIZE = 128
NULL_WEIGHT = 10000


def train_embedding_classifier(
    model_name=None,
    use_metadata_input=False,
    prediction_columns=None,
    pred_weights=None,
    use_class_weights_for_loss=False,
    use_weighted_sampler=False,
    lr=1e-4,
    log=False
):

    train_df = pd.read_parquet(DINO_TRAIN)
    val_df = pd.read_parquet(DINO_VAL)

    # Load it as torch dataset
    train_dataset = EmbeddingMetadataDataset(
        train_df,
        use_input_metadata=use_metadata_input,
        prediction_columns=prediction_columns,
        emb_key="embedding",
    )
    valid_dataset = EmbeddingMetadataDataset(
        val_df,
        use_input_metadata=use_metadata_input,
        prediction_columns=prediction_columns,
        emb_key="embedding",
    )

    prediction_columns = prediction_columns if prediction_columns else ["class_id"]

    col_val_counts = []
    for col in prediction_columns:
        col_val_counts.append(train_df[col].max() + 1)

    idx_splits = np.cumsum(col_val_counts)[:-1].tolist()

    if use_metadata_input:
        INPUT_SIZE = 843  # hard coded magic numbers for now zzz
    else:
        INPUT_SIZE = 768

    model = MultiTargetClassifier(INPUT_SIZE, col_val_counts)

    if use_weighted_sampler:
        c = val_df.class_id.value_counts()
        class_sample_weights = []
        for i in range(1605):
            class_sample_weights.append(1 / c.get(i, NULL_WEIGHT))
        sample_weights = [class_sample_weights[t] for t in train_df.class_id]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )

    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    module = FungiModel(model)
    trainer = L.Trainer(accelerator="gpu", max_epochs=1)

    wandb_logger = WandbLogger(log_model=False, project="FungiClef", name=model_name) if log else None
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="../checkpoints", filename=model_name+"-{epoch}", save_top_k=2, monitor="valid_acc_cls"
    ) if log else None

    trainer = L.Trainer(
        callbacks=[checkpoint_callback], logger=wandb_logger, max_epochs=EPOCHS
    )

    # Use class weight?
    class_weights = None
    if use_class_weights_for_loss:
        class_weights = []
        for col in prediction_columns:

            class_weight = []
            c = val_df[col].value_counts()
            n_class = train_df[col].max() + 1
    
            for i in range(n_class):
                class_weight.append(c.get(i, 0) / c.sum())

            class_weight = torch.Tensor(class_weight)
            
            class_weights.append(class_weight)

    # Try use focal loss
    loss = MultiTargetLoss(idx_splits=idx_splits, class_weights=class_weights, output_weights=pred_weights, device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    multi_output = True if len(prediction_columns) > 1 else False
    module = FungiModel(model, optimizer=optimizer, loss=loss.loss, multi_output=multi_output)

    trainer.fit(module, train_loader, valid_loader)

    wandb.finish()