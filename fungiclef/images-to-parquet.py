# From SnakeCLEF/Murillo Gustinelli
import argparse
import os
from pathlib import Path

from pyspark.sql.functions import element_at, regexp_replace, split

from utils import get_spark

"""
Before running this script, make sure you have downloaded and extracted the dataset into the data folder.
Use the bash file `download_extract_dataset.sh` in the scripts folder.
"""


def create_dataframe(spark, base_dir: Path, raw_root_path: str, meta_dataset_name: str):
    """Converts images into binary data and joins with a Metadata DataFrame"""
    # Load all image files from the base directory as binary data
    image_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(base_dir.as_posix())
    )

    # Construct the string to be replaced - adjust this based on your actual base path
    to_remove = "file:" + str(base_dir.parents[0])

    # Remove "file:{base_dir.parents[0]" from path column
    image_df = image_df.withColumn("path", regexp_replace("path", to_remove, ""))

    # Split the path into an array of elements
    split_path = split(image_df["path"], "/")

    # Extract metadata from the file path
    image_final_df = (
        image_df.withColumn("folder_name", element_at(split_path, -4))
        .withColumn("year", element_at(split_path, -3))
        .withColumn("binomial_name", element_at(split_path, -2))
        .withColumn("file_name", element_at(split_path, -1))
    )

    # Select and rename columns to fit the target schema, including renaming 'content' to 'image_binary_data'
    image_final_df = image_final_df.select(
        "path",
        "folder_name",
        "year",
        "binomial_name",
        "file_name",
        image_final_df["content"].alias("data"),
    )

    # Create a new column "image_path" by removing "/SnakeCLEF2023-small_size/" from "path"
    # This column will be used to join with the metadata df later
    image_final_df = image_final_df.withColumn(
        "image_path", regexp_replace("path", f"^/{base_dir.parts[-1]}/", "")
    )

    # Read the iNaturalist metadata CSV file
    meta_df = spark.read.csv(
        f"{raw_root_path}/{meta_dataset_name}.csv",
        header=True,
        inferSchema=True,
    )

    # Cache the DataFrame to optimize subsequent operations
    meta_df.cache()

    # Drop duplicate entries based on 'image_path' before the join
    meta_df = meta_df.dropDuplicates(["image_path"])

    # Drop 'binomial_name' column since before joining with image_final_df
    meta_final_df = meta_df.drop("binomial_name")

    # Perform an inner join on the 'image_path' column
    final_df = image_final_df.join(meta_final_df, "image_path", "inner")

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
        default="gs://dsgt-clef-fungiclef-2024/data/parquet_files/DF20",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="DF20",
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
    base_dir = Path(args.image_root_path) / "data" / args.dataset_name

    # Create image dataframe
    final_df = create_dataframe(
        spark=spark,
        base_dir=base_dir,
        raw_root_path=args.raw_root_path,
        meta_dataset_name=args.meta_dataset_name,
    )

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(args.output_path)


if __name__ == "__main__":
    main()