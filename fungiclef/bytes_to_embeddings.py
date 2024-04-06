import argparse
import os

from pyspark.ml.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, FloatType
import io
from PIL import Image
import torch
import numpy as np

from utils import get_spark

def make_predict_fn():
    """Return PredictBatchFunction"""
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')

    def predict(inputs: np.ndarray) -> np.ndarray:
        images = [Image.open(io.BytesIO(input)) for input in inputs]
        model_inputs = processor(images=images, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**model_inputs)
            last_hidden_states = outputs.last_hidden_state
        
        numpy_array = last_hidden_states.numpy()
        new_shape = numpy_array.shape[:-2] + (-1,)
        numpy_array = numpy_array.reshape(new_shape)

        return numpy_array

    return predict
    
# batch prediction UDF
apply_dino_pbudf = predict_batch_udf(
    make_predict_fn = make_predict_fn,
    return_type=ArrayType(FloatType()),
    batch_size=8
)

# def create_dataframe(spark, base_dir: Path, raw_root_path: str, meta_dataset_name: str):
def embed_dataframe(spark, size, gcs_parquet_path="gs://dsgt-clef-fungiclef-2024/dev"):
    
    input_folder = f"train/"
    df = spark.read.parquet(gcs_parquet_path+input_folder)

    df_transformed = df.withColumn("transformed_image", apply_dino_pbudf(df["data"]))
    return df_transformed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images and metadata for a dataset stored on GCS."
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="small",
        help="Size of images in dataset, from [small, medium, large]",
    )
    parser.add_argument(
        "--gcs-parquet-path",
        type=str,
        default="gs://dsgt-clef-snakeclef-2024/data/parquet_files/",
        help="GCS path for output Parquet files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Spark
    spark = get_spark(gpu_resources=True, memory='15g', **{
        "spark.sql.parquet.enableVectorizedReader": False, 
    })

    output_folder = f'DINOv2-embeddings-{args.image_size}_size/'
    output_path = args.gcs_parquet_path+output_folder

    # Create image dataframe
    final_df = embed_dataframe(
        spark=spark,
        size=args.image_size,
        gcs_parquet_path=args.gcs_parquet_path
    )

    # Write the DataFrame to GCS in Parquet format
    #final_df.write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    main()