import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyspark.sql.functions as f

from fungiclef.utils import get_spark


def train_test_split(spark, df_path: Path, train_pct: float, stratify_col: str = "class_id"):
    """Given a split fraction converts the big train_test_dataset into two data sets based on the stratify_col"""

    # Load the DataFrame from the Parquet file
    df = spark.read.parquet(df_path)

    # Compute the test fraction and round
    test_pct = np.round(1 - train_pct, 2)

    # Create fractions dict for stratification. The fractions are based on the stratify_col.
    stratify_col_id_list = df.select(stratify_col).rdd.flatMap(lambda x: x).collect()
    train_fraction = {strat_id: train_pct for strat_id in stratify_col_id_list}
    test_fraction = {strat_id: test_pct for strat_id in stratify_col_id_list}

    # Split the DataFrame into train and test DataFrames with fractions dicts
    train_df = df.sampleBy(stratify_col, fractions=train_fraction, seed=42)
    test_df = df.sampleBy(stratify_col, fractions=test_fraction, seed=42)

    # validation
    total_count = df.count()
    train_count = train_df.count()
    test_count = test_df.count()
    print('DataFrame Length:')
    print(f'Train fraction: {train_count/total_count}')
    print(f'Test fraction: {test_count/total_count}')

    df_per_class = df.groupBy(stratify_col).count().withColumnRenamed('count', 'total_count')
    train_per_class = train_df.groupBy(stratify_col).count().withColumnRenamed('count', 'train_count')
    test_per_class = test_df.groupBy(stratify_col).count().withColumnRenamed('count', 'test_count')

    mean_train_frac = train_per_class.join(df_per_class, on=stratify_col, how='inner') \
        .withColumn('train_frac', f.col('train_count') / f.col('total_count')) \
        .agg(f.mean('train_frac')).collect()[0][0]
    
    mean_test_frac = test_per_class.join(df_per_class, on=stratify_col, how='inner') \
        .withColumn('test_frac', f.col('test_count') / f.col('total_count')) \
        .agg(f.mean('test_frac')).collect()[0][0]

    print(f'Avg. fractions per stratify_col: {stratify_col}:')
    print(f'Mean train fraction: {mean_train_frac}')
    print(f'Mean test fraction: {mean_test_frac}')

    return train_df, test_df


def read_config():
    with open('fungiclef/config.json') as f:
        config = json.load(f)
    return config


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    config = read_config()

    # Initialize Spark
    spark = get_spark()

    # Path to the images and metadata in a dataframe of train and test 300px
    df_path = config["gs_paths"]["train_and_test_300px"]["raw_parquet"]


    # Output_paths
    train_output_path = config["gs_paths"]["train_and_test_300px"]["train_parquet"] 
    test_output_path = config["gs_paths"]["train_and_test_300px"]["test_parquet"] 

    # Create image dataframe
    train_df, test_df = train_test_split(
        spark=spark,
        df_path=df_path,
        train_pct=0.8,
    )

    # Write the DataFrame to GCS in Parquet format
    train_df.write.mode("overwrite").parquet(train_output_path)
    test_df.write.mode("overwrite").parquet(test_output_path)


if __name__ == "__main__":
    main()
