# From SnakeCLEF/Murillo Gustinelli
import argparse
import json
from pathlib import Path

from pyspark.sql.functions import element_at, split, regexp_replace

from fungiclef.utils import get_spark

"""
Before running this script, make sure you have downloaded and extracted the dataset into the data folder.
Use the bash file `download_extract_dataset.sh` in the scripts folder.
"""


def create_dataframe(spark, images_path: Path, metadata_path: str):
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
        f"{metadata_path}",
        header=True,
        inferSchema=True,
    )

    # replace .JPG to .jpg
    meta_df = (
        meta_df.withColumn("image_path", regexp_replace('image_path', ".JPG", '.jpg'))
    )

    image_final_df = (
        image_final_df.withColumn("image_path", regexp_replace('image_path', ".JPG", '.jpg'))
    )
    # Perform an inner join on the 'image_path' column
    final_df = image_final_df.join(meta_df, "image_path", "left")

    return final_df


def read_config():
    with open('../config.json') as f:
        config = json.load(f)
    return config


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    config = read_config()

    # Initialize Spark
    spark = get_spark()

    # Set Paths - adjust as needed
    images_path = Path('../../../')  / Path(config["mnt_data_paths"]) / Path("DF20")
    metadata_path = config["gs_paths"]["raw"]["training_metadata"]
    output_path = config["gs_paths"]["parquet"]["training_data"] # here no path

    # Create image dataframe
    final_df = create_dataframe(
        spark=spark,
        images_path=images_path,
        metadata_path=metadata_path,
    )

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    main()