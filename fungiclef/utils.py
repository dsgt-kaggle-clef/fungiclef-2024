# From SnakeCLEF/Murillo Gustinelli
import os
import sys
import json
from contextlib import contextmanager
import torch.nn as nn
import timm
import time

from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(cores=os.cpu_count(),
              memory=os.environ.get("PYSPARK_DRIVER_MEMORY", "4g"),
              executor_memory=os.environ.get("PYSPARK_EXECUTOR_MEMORY", "1g"),
              local_dir="./tmp", app_name="fungi_clef", **kwargs):
    """Get a spark session for a single driver."""
    builder = (
        SparkSession.builder.config("spark.driver.memory", memory)
        .config("spark.jars", "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar")
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.cores", cores)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.local.dir", f"{local_dir}/{int(time.time())}")
    )
    for k, v in kwargs.items():
        builder = builder.config(k, v)
    return builder.appName(app_name).master(f"local[{cores}]").getOrCreate()


@contextmanager
def spark_resource(*args, **kwargs):
    """A context manager for a spark session."""
    spark = None
    try:
        spark = get_spark(*args, **kwargs)
        yield spark
    finally:
        if spark is not None:
            spark.stop()


def read_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    return config

