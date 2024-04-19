## From https://huggingface.co/picekl/FungiCLEF2024-Sample_Submission/blob/main/script.py

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import timm
import torchvision.transforms as T
from PIL import Image
import torch

def is_gpu_available():
    """Check if the python package `onnxruntime-gpu` is installed."""
    return torch.cuda.is_available()


class PytorchWorker:
    """Run inference using ONNX runtime."""

    def __init__(self, model_path: str, model_name: str, number_of_categories: int = 1604):

        def _load_model(model_name, model_path):

            print("Setting up Pytorch Model")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using devide: {self.device}")

            model = timm.create_model(model_name, num_classes=number_of_categories, pretrained=False)

            # if not torch.cuda.is_available():
            #     model_ckpt = torch.load(model_path, map_location=torch.device("cpu"))
            # else:
            #     model_ckpt = torch.load(model_path)

            model_ckpt = torch.load(model_path, map_location=self.device)
            model.load_state_dict(model_ckpt)

            return model.to(self.device).eval()

        self.model = _load_model(model_name, model_path)

        self.transforms = T.Compose([T.Resize((299, 299)),
                                     T.ToTensor(),
                                     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    def predict_image(self, image: np.ndarray) -> list():
        """Run inference using ONNX runtime.
        :param image: Input image as numpy array.
        :return: A list with logits and confidences.
        """

        logits = self.model(self.transforms(image).unsqueeze(0).to(self.device))

        return logits.tolist()


def make_submission(test_metadata, model_path, model_name, output_csv_path="./submission.csv", images_root_path="/tmp/data/private_testset"):
    """Make submission with given """

    model = PytorchWorker(model_path, model_name)

    predictions = []

    for _, row in tqdm(test_metadata.iterrows(), total=len(test_metadata)):
        image_path = os.path.join(images_root_path, row.image_path)

        test_image = Image.open(image_path).convert("RGB")

        logits = model.predict_image(test_image)

        predictions.append(np.argmax(logits))

    test_metadata["class_id"] = predictions

    user_pred_df = test_metadata.drop_duplicates("observation_id", keep="first")
    user_pred_df[["observation_id", "class_id"]].to_csv(output_csv_path, index=None)


if __name__ == "__main__":

    import zipfile

    with zipfile.ZipFile("/tmp/data/private_testset.zip", 'r') as zip_ref:
        zip_ref.extractall("/tmp/data")

    MODEL_PATH = "pytorch_model.bin"
    MODEL_NAME = "tf_efficientnet_b1.ap_in1k"

    metadata_file_path = "./FungiCLEF2024_TestMetadata.csv"
    test_metadata = pd.read_csv(metadata_file_path)

    make_submission(
        test_metadata=test_metadata,
        model_path=MODEL_PATH,
        model_name=MODEL_NAME
    )
