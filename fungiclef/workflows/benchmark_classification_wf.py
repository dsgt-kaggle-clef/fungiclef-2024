from argparse import ArgumentParser

import luigi
import torch
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import functions as F

class BenchmarkClassificationWorkflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-fungiclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/processed/metadata_clean/v1",
            dst_path=f"{self.local_root}/processed/metadata_clean/v1",
        )

        for k in [3, 20, 100, 200]:
            yield [
                # these runs are meant to validate that the pipeline works as expected before expensive runs
                *(
                    [
                        # non-linear scaling behavior for some reason, this
                        # is terrible as a baseline because it literally takes forever
                        FitLogisticModel(
                            k=k,
                            shuffle_partitions=32,
                            input_path=f"{self.local_root}/processed/metadata_clean/v1",
                            output_path=f"{self.local_root}/models/benchmark_classification/v2/model=logistic/k={k}",
                        ),
                        # unfortunately this only supports up to 100 classes, but
                        # otherwise the performance is great
                        FitRandomForestModel(
                            k=k,
                            shuffle_partitions=32,
                            input_path=f"{self.local_root}/processed/metadata_clean/v1",
                            output_path=f"{self.local_root}/models/benchmark_classification/v2/model=random_forest/k={k}",
                        ),
                    ]
                    if k <= 100
                    else []
                ),
                # multiple workers for xgboost because a single worker is deadfully slow
                *[
                    *[
                        FitXGBoostModel(
                            k=k,
                            shuffle_partitions=32,
                            num_workers=num_workers,
                            input_path=f"{self.local_root}/processed/metadata_clean/v1",
                            output_path=f"{self.local_root}/models/benchmark_classification/v2/model=xgboost_n{num_workers}/k={k}",
                        )
                        for num_workers in [2, 8, 16]
                    ],
                    *(
                        [
                            FitXGBoostModel(
                                k=k,
                                shuffle_partitions=32,
                                num_workers=num_workers,
                                device="gpu",
                                input_path=f"{self.local_root}/processed/metadata_clean/v1",
                                output_path=f"{self.local_root}/models/benchmark_classification/v2/model=xgboost_gpu_n{num_workers}/k={k}",
                            )
                            # we can only have one worker per gpu
                            for num_workers in [1]
                        ]
                        if torch.cuda.is_available()
                        else []
                    ),
                ],
            ]

        # now print the results
        with spark_resource() as spark:
            df = (
                spark.read.json(
                    f"{self.local_root}/models/benchmark_classification/v2/*/*/perf"
                )
                .withColumn("file", F.input_file_name())
                .select(
                    F.regexp_extract("file", r"model=(\w+)", 1).alias("model"),
                    F.regexp_extract("file", r"k=(\d+)", 1).cast("integer").alias("k"),
                    F.round(F.col("avg_metrics")[0], 2).alias("f1_avg"),
                    F.round(F.col("std_metrics")[0], 2).alias("f1_std"),
                    F.round("train_time", 2).alias("train_time"),
                )
            ).orderBy("model", "k")
            df.show(n=100)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [BenchmarkClassificationWorkflow()],
        scheduler_host=args.scheduler_host,
        workers=1,
    )