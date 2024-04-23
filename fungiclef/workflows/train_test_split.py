"""MAPS unknown class_id from -1 to max() +1
and splits the dataset into train and test sets based on the class_id column
"""

import numpy as np
import pyspark.sql.functions as f
import pyspark
from fungiclef.utils import get_spark, read_config


def train_test_split(df: pyspark.sql.DataFrame, train_pct: float, stratify_col: str = "class_id"):
    """
    Splits a DataFrame into train and test DataFrames based on a given split fraction and a stratification column.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame to be split.
        train_pct (float): The fraction of data to be used for training. Should be between 0 and 1.
        stratify_col (str, optional): The column used for stratification. Defaults to "class_id".

    Returns:
        tuple: A tuple containing the train DataFrame and the test DataFrame.
    """

    # Compute the test fraction and round
    test_pct = np.round(1 - train_pct, 2)

    print(f"Splitting the dataset into {train_pct} train and {test_pct} test based on {stratify_col}")

    # Create fractions dict for stratification. The fractions are based on the stratify_col.
    stratify_col_id_list = df.select(stratify_col).rdd.flatMap(lambda x: x).collect()
    train_fraction = {strat_id: train_pct for strat_id in stratify_col_id_list}

    # Create identifier col to filter out the train data and create the test data after the train data is filtered out
    df = df.withColumn("identifier", f.monotonically_increasing_id())

    # Split the DataFrame into train and test DataFrames with fractions dicts
    train_df = df.sampleBy(stratify_col, fractions=train_fraction, seed=42)

    # Filter out the train data from the original DataFrame to create the test DataFrame
    test_df = df.join(train_df, on="identifier", how="left_anti").drop("identifier")
    train_df = train_df.drop("identifier")

    # validation
    total_count = df.count()
    train_count = train_df.count()
    test_count = test_df.count()
    print("DataFrame Length:")
    print(f"Train fraction: {train_count/total_count}")
    print(f"Test fraction: {test_count/total_count}")
    print("Absolute numbers")
    print(f"Total count: {total_count}")
    print(f"Train count: {train_count}")
    print(f"Test count: {test_count}")

    df_per_class = df.groupBy(stratify_col).count().withColumnRenamed("count", "total_count")
    train_per_class = train_df.groupBy(stratify_col).count().withColumnRenamed("count", "train_count")
    test_per_class = test_df.groupBy(stratify_col).count().withColumnRenamed("count", "test_count")

    mean_train_frac = (
        train_per_class.join(df_per_class, on=stratify_col, how="inner")
        .withColumn("train_frac", f.col("train_count") / f.col("total_count"))
        .agg(f.mean("train_frac"))
        .collect()[0][0]
    )

    mean_test_frac = (
        test_per_class.join(df_per_class, on=stratify_col, how="inner")
        .withColumn("test_frac", f.col("test_count") / f.col("total_count"))
        .agg(f.mean("test_frac"))
        .collect()[0][0]
    )

    print(f"Avg. fractions per stratify_col {stratify_col}:")
    print(f"Mean train fraction: {mean_train_frac}")
    print(f"Mean test fraction: {mean_test_frac}")

    return train_df, test_df


def manipulate_class_id(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Manipulates the class_id column in the given DataFrame.
    Unknown (-1) are mapped to max() + 1.
    Test data (NaN) is mapped to max() + 2.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame containing the class_id column.

    Returns:
        pyspark.sql.DataFrame: The modified DataFrame with the class_id column manipulated.
    """

    # Map unknown class_id from -1 to max() + 1
    unknown_class_id_count = df.filter(f.col("class_id") == -1).count()
    max_class_id = df.agg({"class_id": "max"}).collect()[0][0]
    df = df.withColumn(
        "class_id",
        f.when(f.col("class_id") == -1, max_class_id + 1).otherwise(f.col("class_id")),
    )
    print(f"{unknown_class_id_count} unknown class_id examples mapped to {max_class_id + 1}")

    # TODO: What to do with the public_test_metadata?
    # Add unknown from public_test_metadata to another unknown class_id (max() + 2)
    test_class_id_count = df.filter(f.col("data_set") == "test").count()
    df = df.withColumn(
        "class_id",
        f.when(f.col("data_set") == "test", max_class_id + 2).otherwise(f.col("class_id")),
    )
    print(f"{test_class_id_count} test class_id examples mapped to {max_class_id + 2}")

    return df


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    config = read_config("fungiclef/config.json")

    # Initialize Spark
    spark = get_spark()

    # Path to the images and metadata in a dataframe of train and test 300px
    df_path = config["gs_paths"]["train_and_test_300px_corrected"]["raw_parquet"]

    # Output_paths
    train_output_path = config["gs_paths"]["train_and_test_300px_corrected"]["train_parquet"]
    test_output_path = config["gs_paths"]["train_and_test_300px_corrected"]["test_parquet"]

    # Load the DataFrame from the Parquet file
    df = spark.read.parquet(df_path)

    # Prep class_id column
    df = manipulate_class_id(df)

    # Create image dataframe
    train_df, test_df = train_test_split(
        df=df,
        train_pct=0.9,
        stratify_col="class_id",
    )

    # Write the DataFrame to GCS in Parquet format
    train_df.write.mode("overwrite").parquet(train_output_path)
    test_df.write.mode("overwrite").parquet(test_output_path)

    print("Dataframes written to GCS in Parquet format.")
    print(f"Train: {train_output_path}")
    print(f"Test: {test_output_path}")


if __name__ == "__main__":
    main()
