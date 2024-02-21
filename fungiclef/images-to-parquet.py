# From SnakeCLEF/Murillo Gustinelli
import argparse
import os
from pathlib import Path

from pyspark.sql.functions import element_at, split

from utils import get_spark

"""
Before running this script, make sure you have downloaded and extracted the dataset into the data folder.
Use the bash file `download_extract_dataset.sh` in the scripts folder.
"""


def create_dataframe(spark, images_path: Path, raw_root_path: str, meta_data_path: str):
    """Converts images into binary data and joins with a Metadata DataFrame"""
    # Load all image files from the base directory as binary data
    image_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .load(images_path.as_posix())
    )

    # Split the path into an array of elements
    split_path = split(image_df["path"], "/")

    # Extract metadata from the file path
    image_final_df = (
        image_df.withColumn("image_path", element_at(split_path, -1))
    )

    # Select and rename columns to fit the target schema, including renaming 'content' to 'image_binary_data'
    image_final_df = image_final_df.select(
        "image_path",
        image_final_df["content"].alias("data"),
    )

    # Read the metadata CSV file & cache file
    meta_df = spark.read.csv(
        f"{meta_data_path}.csv",
        header=True,
        inferSchema=True,
    )

    # No duplicates or columns to drop

    # Perform an inner join on the 'image_path' column
    final_df = image_final_df.join(meta_df, "image_path", "inner")

    return final_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images and metadata for a dataset stored on GCS."
    )
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=str(Path(".").anchor / Path("mnt/")),
        help="Base directory path for image data",
    )
    parser.add_argument(
        "--raw-root-path",
        type=str,
        default="gs://dsgt-clef-fungiclef-2024/raw/",
        help="Root directory path for metadata",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="gs://dsgt-clef-fungiclef-2024/data/parquet/DF20",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="DF20_300",
        help="Dataset name downloaded from tar file",
    )
    parser.add_argument(
        "--meta-dataset-name",
        type=str,
        default="FungiCLEF2023_train_metadata_PRODUCTION",
        help="Train Metadata CSV file",
    )

    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    args = parse_args()

    # Initialize Spark
    spark = get_spark()

    # Convert image-root-path to a Path object here
    images_path = Path('../../../')  / Path(args.image_root_path) / "data" / args.dataset_name

    meta_data_path = Path('../../../')  / Path(args.image_root_path) / "data" / args.meta_dataset_name

    # Create image dataframe
    final_df = create_dataframe(
        spark=spark,
        images_path=images_path,
        raw_root_path=args.raw_root_path,
        meta_data_path=meta_data_path,
    )

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(args.output_path)


if __name__ == "__main__":
    main()