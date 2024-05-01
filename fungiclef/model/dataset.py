from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import io
import torch
import cv2


class ImageDataset(Dataset):
    def __init__(self, df, transform=None, local_filepath=None):
        self.df = df
        self.transform = transform
        self.local_filepath = local_filepath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df["class_id"].values[idx]

        if self.local_filepath:
            file_path = self.local_filepath + self.df["image_path"].values[idx]

            image = cv2.imread(file_path)

            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image = np.random.uniform(-1, 1, size=(299, 299, 3)).astype(np.float32)
                print(file_path)

        else:

            image = Image.open(io.BytesIO(self.df.data.values[idx]))
            image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


class EmbeddingDataset(Dataset):
    def __init__(self, df, emb_key="embeddings"):
        self.df = df
        self.emb_key = emb_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df["class_id"].values[idx]

        embedding = np.array(self.df[self.emb_key].values[idx]).astype(np.float32)

        return embedding, label


INPUT_METADATA_COLUMNS = [
    "month",
    "geohash_int_normalized",
    "substrate_0",
    "substrate_1",
    "substrate_2",
    "substrate_3",
    "substrate_4",
    "substrate_5",
    "substrate_6",
    "substrate_7",
    "substrate_8",
    "substrate_9",
    "substrate_10",
    "substrate_11",
    "substrate_12",
    "substrate_13",
    "substrate_14",
    "substrate_15",
    "substrate_16",
    "substrate_17",
    "substrate_18",
    "substrate_19",
    "substrate_20",
    "substrate_21",
    "substrate_22",
    "substrate_23",
    "substrate_24",
    "substrate_25",
    "substrate_26",
    "substrate_27",
    "substrate_28",
    "substrate_29",
    "substrate_30",
    "metasubstrate_0",
    "metasubstrate_1",
    "metasubstrate_2",
    "metasubstrate_3",
    "metasubstrate_4",
    "metasubstrate_5",
    "metasubstrate_6",
    "metasubstrate_7",
    "metasubstrate_8",
    "metasubstrate_9",
    "habitat_0",
    "habitat_1",
    "habitat_2",
    "habitat_3",
    "habitat_4",
    "habitat_5",
    "habitat_6",
    "habitat_7",
    "habitat_8",
    "habitat_9",
    "habitat_10",
    "habitat_11",
    "habitat_12",
    "habitat_13",
    "habitat_14",
    "habitat_15",
    "habitat_16",
    "habitat_17",
    "habitat_18",
    "habitat_19",
    "habitat_20",
    "habitat_21",
    "habitat_22",
    "habitat_23",
    "habitat_24",
    "habitat_25",
    "habitat_26",
    "habitat_27",
    "habitat_28",
    "habitat_29",
    "habitat_30",
    "habitat_31",
]


class EmbeddingMetadataDataset(Dataset):
    def __init__(
        self,
        df,
        emb_key="embedding",
        use_input_metadata=False,
        prediction_columns=None,
    ):
        self.df = df
        self.emb_key = emb_key
        self.use_input_metadata = use_input_metadata    
        
        prediction_columns = prediction_columns if prediction_columns else ['class_id']

        if use_input_metadata:
            self.inputs = np.concatenate((np.array(df[emb_key].tolist()), df[INPUT_METADATA_COLUMNS].to_numpy()), axis=1).astype(np.float32)
        else:
            self.inputs = np.array(df[emb_key].tolist()).astype(np.float32)

        self.labels = df[prediction_columns].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        return self.inputs[idx], self.labels[idx]
