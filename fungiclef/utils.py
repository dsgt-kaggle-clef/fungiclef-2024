# From SnakeCLEF/Murillo Gustinelli
import os
import sys
import json
from contextlib import contextmanager
from textwrap import dedent
import luigi.contrib.gcs
import tempfile
from luigi.contrib.external_program import ExternalProgramTask

from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(cores=4, memory="8g", local_dir="/mnt/data/tmp", **kwargs):
    """Get a spark session for a single driver."""
    builder = (
        SparkSession.builder.config("spark.driver.memory", memory)
        .config("spark.driver.cores", cores)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.local.dir", local_dir)
    )
    for k, v in kwargs.items():
        builder = builder.config(k, v)
    return builder.getOrCreate()


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


def maybe_gcs_target(path: str) -> luigi.Target:
    """Return a GCS target if the path starts with gs://, otherwise a LocalTarget."""
    if path.startswith("gs://"):
        return luigi.contrib.gcs.GCSTarget(path)
    else:
        return luigi.LocalTarget(path)
    

class BashScriptTask(ExternalProgramTask):
    def script_text(self) -> str:
        """The contents of to write to a bash script for running."""
        return dedent(
            """
            #!/bin/bash
            echo 'hello world'
            exit 1
            """
        )

    def program_args(self):
        """Execute the script."""
        script_text = self.script_text().strip()
        script_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        script_file.write(script_text)
        script_file.close()
        print(f"Script file: {script_file.name}")
        print(script_text)
        return ["/bin/bash", script_file.name]
    

class RsyncGCSFiles(BashScriptTask):
    """Download using the gcloud command-line tool."""

    src_path = luigi.Parameter()
    dst_path = luigi.Parameter()

    def output(self):
        path = f"{self.dst_path}/_SUCCESS"
        if path.startswith("gs://"):
            return luigi.contrib.gcs.GCSTarget(path)
        else:
            return luigi.LocalTarget(path)

    def script_text(self) -> str:
        return dedent(
            f"""
            #!/bin/bash
            set -eux -o pipefail
            gcloud storage rsync -r {self.src_path} {self.dst_path}
            """
        )

    def run(self):
        super().run()
        with self.output().open("w") as f:
            f.write("")