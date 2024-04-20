import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from fungiclef.embedding.transforms import DCTN, WrappedDinoV2
from fungiclef.utils import spark_resource


class ProcessBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    should_subset = luigi.BoolParameter(default=False)
    num_partitions = luigi.IntParameter(default=100)
    sample_id = luigi.OptionalIntParameter(
        default=None
    )  # helper column to split the data
    num_sample_id = luigi.IntParameter(
        default=10
    )  # Split the DataFrame and transformations into batches

    def output(self):
        if self.sample_id is None:  # If not using subset
            # save both the model pipeline and the dataset
            return luigi.contrib.gcs.GCSTarget(
                f"{self.output_path}/data/_SUCCESS"
            )
        else:
            return luigi.contrib.gcs.GCSTarget(
                f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS"
            )

    @property
    def feature_columns(self) -> list:
        raise NotImplementedError()

    def pipeline(self) -> Pipeline:
        raise NotImplementedError()

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)

        if self.sample_id is not None:
            transformed = (
                transformed.withColumn(
                    "sample_id",
                    F.crc32(F.col("species").cast("string"))
                    % self.num_sample_id,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )

        for c in features:
            # check if the feature is a vector and convert it to an array
            if "array" in transformed.schema[c].simpleString():
                continue
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

    def _get_subset(self, df):
        # Get subset of images to test pipeline

        subset_df = df.limit(200).cache()

        return subset_df

    def run(self):
        with spark_resource(
            **{"spark.sql.shuffle.partitions": self.num_partitions,
               "spark.sql.parquet.enableVectorizedReader": False}
        ) as spark:
            df = spark.read.parquet(self.input_path)

            if self.should_subset:
                # Get subset of data to test pipeline
                df = self._get_subset(df=df)

            model = self.pipeline().fit(df)
            model.write().overwrite().save(f"{self.output_path}/model")
            transformed = self.transform(model, df, self.feature_columns)

            if self.sample_id is None:
                output_path = f"{self.output_path}/data"
            else:
                output_path = (
                    f"{self.output_path}/data/sample_id={self.sample_id}"
                )

            print(f"Writing to {output_path}")
            transformed.repartition(self.num_partitions).write.mode(
                "overwrite"
            ).parquet(output_path)


class ProcessDino(ProcessBase):
    @property
    def feature_columns(self):
        return ["dino_embedding"]

    def pipeline(self):
        dino = WrappedDinoV2(input_col="data", output_col="dino_embedding")
        return Pipeline(
            stages=[
                dino,
                SQLTransformer(
                    statement=f"SELECT ImageUniqueID, species, dino_embedding FROM __THIS__"
                ),
            ]
        )


class ProcessDCT(ProcessBase):
    @property
    def feature_columns(self):
        return ["dct_embedding"]

    def pipeline(self):
        dct = DCTN(input_col="dino_embedding", output_col="dct_embedding")
        return Pipeline(
            stages=[
                dct,
                SQLTransformer(
                    statement=f"SELECT ImageUniqueID, species, dct_embedding FROM __THIS__"
                ),
            ]
        )


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        # Run jobs with subset and full-size data
        for subset in [True, False]:
            final_output_path = self.output_path
            if subset:
                subset_path = f"subset_{self.output_path.split('/')[-1]}"
                final_output_path = self.output_path.replace(
                    self.output_path.split("/")[-1], subset_path
                )

            yield ProcessDino(
                    input_path=self.input_path,
                    output_path=f"{final_output_path}/dino",
                    should_subset=subset,
                    sample_id=None,
                    num_sample_id=10,
                )

            yield ProcessDCT(
                input_path=f"{final_output_path}/dino/data",
                output_path=f"{final_output_path}/dino_dct",
                should_subset=subset,
            )


def run_embeddings_wf(input_path: str, output_path: str):

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                output_path=output_path,
            )
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
