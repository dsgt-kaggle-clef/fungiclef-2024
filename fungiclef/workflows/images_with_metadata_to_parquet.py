# From SnakeCLEF/Murillo Gustinelli
import json
from pathlib import Path

from pyspark.sql.functions import element_at, split, regexp_replace, lit

from fungiclef.utils import get_spark

"""
Before running this script, make sure you have downloaded and extracted the dataset into the data folder.
Use the bash file `download_extract_dataset.sh` in the scripts folder.
"""


def create_dataframe(spark, images_path: dict[str, Path], metadata_path: dict[str, Path]):
    """Converts images into binary data and joins with a Metadata DataFrame"""
    # Load all image files from the base directory as binary data

    image_df_dict = dict()
    img_stats_dict = dict()
    for name, image_path in images_path.items():
        # train and val have slightly different extensions
        if name == "train":
            extension = "*.jpg"
        else:
            extension = "*.JPG"

        image_df = spark.read.format("binaryFile").option("pathGlobFilter", extension).load(image_path.as_posix())

        # Split the path into an array of elements
        split_path = split(image_df["path"], "/")

        # Extract image_path only from the file path and discard dir info
        image_final_df = image_df.withColumn("image_path", element_at(split_path, -1))

        # Select and rename columns to fit the target schema, including renaming 'content' to 'image_binary_data'
        image_final_df = image_final_df.select(
            "image_path",
            image_final_df["content"].alias("data"),
        )

        image_final_df = image_final_df.withColumn("image_path", regexp_replace("image_path", ".JPG", ".jpg"))

        length_df = image_final_df.count()
        img_stats_dict.update({name: length_df})
        image_df_dict.update({name: image_final_df})

    metadata_df_dict = dict()
    meta_stats_dict = dict()
    for name, meta_path in metadata_path.items():
        # Read the metadata CSV file & cache file
        meta_df = spark.read.csv(
            f"{meta_path}",
            header=True,
            inferSchema=True,
        )

        # here we need to replace all .JPG to .jpg
        meta_df = meta_df.withColumn("image_path", regexp_replace("image_path", ".JPG", ".jpg"))

        # indicator col to know where data comes from
        meta_df = meta_df.withColumn("data_set", lit(f"{name}"))

        length_df = meta_df.count()
        meta_stats_dict.update({name: length_df})
        metadata_df_dict.update({name: meta_df})

    # Combine Val and Test for easier processing with image_df_dict
    if metadata_df_dict.get("test"):
        metadata_df_dict["val"] = metadata_df_dict["val"].unionByName(metadata_df_dict.pop("test"), allowMissingColumns=True)
        meta_stats_dict["val_and_test"] = meta_stats_dict["val"] + meta_stats_dict["test"]

    # Perform an inner join on the 'image_path' column
    final_df_dict = dict()
    for name, meta_path in images_path.items():
        image_final_df = image_df_dict[name]
        meta_df = metadata_df_dict[name]
        final_df = image_final_df.join(meta_df, "image_path", "left")
        final_df_dict.update({name: final_df})

    final_df = final_df_dict["train"].unionByName(final_df_dict["val"], allowMissingColumns=True)

    final_merge_stats = final_df.groupBy("data_set").count()
    print("After merging metadata on images.")
    print("Final merged stats:")
    print(final_merge_stats.show())
    print("Corresponds to the sum of the individual stats:")
    print("Image stats:")
    print(img_stats_dict)
    print("Metadata stats:")
    print(meta_stats_dict)
    return final_df


def read_config():
    with open("fungiclef/config.json") as f:
        config = json.load(f)
    return config


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    config = read_config()

    # Initialize Spark
    spark = get_spark()

    # Train data
    images_train_path = Path("../../../") / Path(config["mnt_data_paths"]) / Path("DF20_300")
    metadata_train_path = config["gs_paths"]["train"]["metadata"]

    # Val data
    images_val_path = Path("../../../") / Path(config["mnt_data_paths"]) / Path("DF21_300")
    metadata_val_path = config["gs_paths"]["val"]["metadata"]

    # Test data
    metadata_test_path = config["gs_paths"]["test"]["metadata"]

    # Output path
    output_path = config["gs_paths"]["train_and_test_300px_corrected"]["raw_parquet"]

    images_paths = {"train": images_train_path, "val": images_val_path}
    metadata_paths = {
        "train": metadata_train_path,
        "val": metadata_val_path,
        # "test": metadata_test_path,
    }

    # Create image dataframe
    final_df = create_dataframe(
        spark=spark,
        images_path=images_paths,
        metadata_path=metadata_paths,
    )

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(output_path)

    print("Dataframe written to GCS in Parquet format.")


if __name__ == "__main__":
    main()
