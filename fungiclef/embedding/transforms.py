import io

import numpy as np
import torch
from PIL import Image
from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasInputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType
from scipy.fftpack import dctn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
)


class WrappedDinoV2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for DinoV2 to add it to the pipeline
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        model_name="facebook/dinov2-base",
        batch_size=8,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.model_name = model_name
        self.batch_size = batch_size

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)

        # Move model to GPU
        if torch.cuda.is_available():
            model = model.to("cuda")

        def predict(inputs: np.ndarray) -> np.ndarray:
            images = [Image.open(io.BytesIO(input)) for input in inputs]
            model_inputs = processor(images=images, return_tensors="pt")

            # Move inputs to GPU
            if torch.cuda.is_available():
                model_inputs = {
                    key: value.to("cuda")
                    for key, value in model_inputs.items()
                }

            with torch.no_grad():
                outputs = model(**model_inputs)
                last_hidden_states = outputs.last_hidden_state

            numpy_array = last_hidden_states.cpu().numpy()
            new_shape = numpy_array.shape[:-2] + (-1,)
            numpy_array = numpy_array.reshape(new_shape)

            return numpy_array

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.batch_size,
            )(self.getInputCol()),
        )


class DCTN(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Run n-dimensional DCT on the input column
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        filter_size: int = 8,
        batch_size=8,
        input_tensor_shapes=[[257, 768]],
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.input_tensor_shapes = input_tensor_shapes

    def _make_predict_fn(self):
        def dctn_filter(tile, k):
            coeff = dctn(tile)
            coeff_subset = coeff[:k, :k]
            return coeff_subset.flatten()

        def predict(inputs: np.ndarray) -> np.ndarray:
            # inputs is a 3D array of shape (batch_size, img_dim, img_dim)
            return np.array(
                [dctn_filter(x, k=self.filter_size) for x in inputs]
            )

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.batch_size,
                input_tensor_shapes=self.input_tensor_shapes,
            )(self.getInputCol()),
        )


class WrappedCLIP(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for CLIP to add it to the pipeline
    """

    def __init__(
        self,
        input_cols: list[str] = ["input_img", "input_txt"],
        output_col: str = "output",
        model_name="openai/clip-vit-base-patch32",
        batch_size=64,
    ):
        super().__init__()
        self._setDefault(inputCols=input_cols, outputCol=output_col)
        self.model_name = model_name
        self.batch_size = batch_size

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        processor = CLIPProcessor.from_pretrained(self.model_name)
        model = CLIPModel.from_pretrained(self.model_name)
        
        # Move model to GPU
        if torch.cuda.is_available():
            model = model.to("cuda")

        def predict(images: np.ndarray, texts: np.ndarray) -> np.ndarray:
            image_obj = [Image.open(io.BytesIO(img)) for img in images]

            #print(texts)
            text = list()
            for txt in texts:
                text.append(txt)
            #print(text)

            #text = texts
            model_inputs = processor(
                text=text, images=image_obj, return_tensors="pt", padding=True, truncation=True
            )

            # Move inputs to GPU
            if torch.cuda.is_available():
                model_inputs = {
                    key: value.to("cuda")
                    for key, value in model_inputs.items()
                }

            with torch.no_grad():
                outputs = model(**model_inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

            # Concatenate or otherwise combine image and text features
            combined_features = torch.cat((image_features, text_features), dim=1)
            numpy_array = combined_features.cpu().numpy()

            return numpy_array

        return predict

    def _transform(self, df: DataFrame):
        predict_udf = predict_batch_udf(self._make_predict_fn, return_type=ArrayType(FloatType()), batch_size=self.batch_size)
        return df.withColumn(self.getOutputCol(), predict_udf(self.getInputCols()[0], self.getInputCols()[1]))

