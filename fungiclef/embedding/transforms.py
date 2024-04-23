import io

import numpy as np
import torch
from PIL import Image
from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType
from scipy.fftpack import dctn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    CLIPImageProcessor,
    CLIPTokenizer,
)


class HasFilterSize(Params):
    filter_size = Param(
        Params._dummy(),
        "filter_size",
        "filter size to use for DCTN",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(filter_size=8)

    def getFilterSize(self):
        return self.getOrDefault(self.filter_size)


class HasInputTensorShapes(Params):
    input_tensor_shapes = Param(
        Params._dummy(),
        "input_tensor_shapes",
        "shape of the tensor",
        typeConverter=TypeConverters.toListInt,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(input_tensor_shapes=[8])

    def getInputTensorShapes(self):
        return self.getOrDefault(self.input_tensor_shapes)


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
                model_inputs = {key: value.to("cuda") for key, value in model_inputs.items()}

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
    HasFilterSize,
    HasInputTensorShapes,
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
        input_tensor_shapes=[257, 768],
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            filter_size=filter_size,
            input_tensor_shapes=input_tensor_shapes,
        )
        self.batch_size = batch_size

    def _make_predict_fn(self):
        def dctn_filter(tile, k):
            coeff = dctn(tile)
            coeff_subset = coeff[:k, :k]
            return coeff_subset.flatten()

        def predict(inputs: np.ndarray) -> np.ndarray:
            # inputs is a 3D array of shape (batch_size, img_dim, img_dim)
            return np.array([dctn_filter(x, k=self.getFilterSize()) for x in inputs])

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.batch_size,
                input_tensor_shapes=[self.getInputTensorShapes()],
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
        output_col_dummy: str = "output",
        output_cols: list[str] = ["image", "text", "dot", "similarity"],
        model_name="openai/clip-vit-base-patch32",
        batch_size=64,
    ):
        super().__init__()
        self._setDefault(inputCols=input_cols, outputCol=output_col_dummy)
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_cols = output_cols

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # cannot use unk and pad token yet because I need to include it vocab as well
        tokenizer = CLIPTokenizer.from_pretrained(
            self.model_name,
            padding=True,
            truncation=True,
            # unknown_token="<|unk|>",
            # pad_token="<|pad|>",
            return_tensors="pt",
            max_length=77,
        )

        image_processor = CLIPImageProcessor.from_pretrained(
            self.model_name,
            return_tensors="pt",
        )
        processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
        model = CLIPModel.from_pretrained(self.model_name)

        # Move model to GPU
        if torch.cuda.is_available():
            model = model.to("cuda")

        def predict(images: np.ndarray, texts: np.ndarray) -> np.ndarray:
            image_obj = [Image.open(io.BytesIO(img)) for img in images]

            text_obj = [txt for txt in texts]

            model_inputs = processor(
                text=text_obj,
                images=image_obj,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Move inputs to GPU
            if torch.cuda.is_available():
                model_inputs = {key: value.to("cuda") for key, value in model_inputs.items()}

            with torch.no_grad():
                outputs = model(**model_inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                # Combine image and text features into an embedding which accounts for similiarity of both (dot product)
                dot_product = image_features @ text_features.T
                similarity = outputs.logits_per_image

            return {
                self.output_cols[0]: image_features.cpu().numpy(),
                self.output_cols[1]: text_features.cpu().numpy(),
                self.output_cols[2]: dot_product.cpu().numpy(),
                self.output_cols[3]: similarity.cpu().numpy(),
            }

        return predict

    def _transform(self, df: DataFrame):
        # predict_udf requires return_type. My CLIP predict function returns a dictionary with 3 keys
        # This can be stored in a pyspark StructField
        schema = StructType(
            [
                StructField(self.output_cols[0], ArrayType(FloatType()), True),
                StructField(self.output_cols[1], ArrayType(FloatType()), True),
                StructField(self.output_cols[2], ArrayType(FloatType()), True),
                StructField(self.output_cols[3], ArrayType(FloatType()), True),
            ]
        )

        predict_udf = predict_batch_udf(self._make_predict_fn, return_type=schema, batch_size=self.batch_size)

        # perform predict batch udf and save StructField Result in dummy output COL
        df = df.withColumn(
            self.getOutputCol(),
            predict_udf(self.getInputCols()[0], self.getInputCols()[1]),
        )

        # get the results of the StructField and save them in the real outputcols
        return (
            df.withColumn(self.output_cols[0], df[self.getOutputCol()][self.output_cols[0]])
            .withColumn(self.output_cols[1], df[self.getOutputCol()][self.output_cols[1]])
            .withColumn(self.output_cols[2], df[self.getOutputCol()][self.output_cols[2]])
            .withColumn(self.output_cols[3], df[self.getOutputCol()][self.output_cols[3]])
            .drop(self.getOutputCol())
        )
