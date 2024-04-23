import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from fungiclef.embedding.transforms import DCTN, WrappedDinoV2, WrappedCLIPV2
from fungiclef.utils import spark_resource


class ProcessBase(luigi.Task):
    """
    Base class for processing tasks.

    Attributes:
        input_path (str): The path to the input data.
        output_path (str): The path to save the output data.
        should_subset (bool): Flag indicating whether to use a subset of the data.
        num_partitions (int): The number of partitions to use for data processing.
        sample_id (Optional[int]): The sample ID to filter the data by.
        num_sample_id (int): The number of sample IDs to split the data into batches.
    """

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    should_subset = luigi.BoolParameter(default=False)
    num_partitions = luigi.IntParameter(default=100)
    sample_id = luigi.OptionalIntParameter(default=None)  # helper column to split the data
    num_sample_id = luigi.IntParameter(default=10)  # Split the DataFrame and transformations into batches

    def output(self):
        if self.sample_id is None:  # If not using subset
            # save both the model pipeline and the dataset
            return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/data/_SUCCESS")
        else:
            return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS")

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
                    F.crc32(F.col("species").cast("string")) % self.num_sample_id,
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

    def _preprocess(self, df) -> DataFrame:
        return df

    def run(self):
        with spark_resource(
            **{
                "spark.sql.shuffle.partitions": self.num_partitions,
                "spark.sql.parquet.enableVectorizedReader": False,
            }
        ) as spark:
            df = spark.read.parquet(self.input_path)

            if self.should_subset:
                # Get subset of data to test pipeline
                df = self._get_subset(df=df)

            df = self._preprocess(df)

            model = self.pipeline().fit(df)
            model.write().overwrite().save(f"{self.output_path}/model")
            transformed = self.transform(model, df, self.feature_columns)

            if self.sample_id is None:
                output_path = f"{self.output_path}/data"
            else:
                output_path = f"{self.output_path}/data/sample_id={self.sample_id}"

            print(f"Writing to {output_path}")
            transformed.repartition(self.num_partitions).write.mode("overwrite").parquet(output_path)


class ProcessDino(ProcessBase):
    """
    A class representing the process for generating Dino embeddings.

    Attributes:
        feature_columns (list): A list of feature columns used in the pipeline.

    Methods:
        pipeline(): Returns the pipeline for generating Dino embeddings.
    """

    @property
    def feature_columns(self):
        return ["dino_embedding"]

    def pipeline(self):
        """
        Returns the pipeline for generating Dino embeddings.

        Returns:
            Pipeline: The pipeline object containing the necessary stages.
        """
        dino = WrappedDinoV2(input_col="data", output_col="dino_embedding")
        return Pipeline(
            stages=[
                dino,
                SQLTransformer(statement="SELECT ImageUniqueID, species, dino_embedding FROM __THIS__"),
            ]
        )


class ProcessDCTN(ProcessBase):
    """
    A class representing a data processing task using DCTN embeddings.

    Attributes:
        filter_size (int): The size of the filter used in the DCTN algorithm.
    """

    filter_size = luigi.IntParameter(default=8)

    @property
    def feature_columns(self):
        return ["dct_embedding"]

    def pipeline(self):
        """
        Defines the data processing pipeline using DCTN embeddings.

        Returns:
            Pipeline: The data processing pipeline.
        """
        dct = DCTN(
            input_col="dino_embedding",
            output_col="dct_embedding",
            filter_size=self.filter_size,
        )
        return Pipeline(
            stages=[
                dct,
                SQLTransformer(statement="SELECT ImageUniqueID, species, dct_embedding FROM __THIS__"),
            ]
        )


class ProcessCLIP(ProcessBase):
    """
    A class for processing CLIP data.

    Attributes:
        concat_text_data (list): A list of text columns to concatenate.

    Methods:
        feature_columns: Returns the list of feature columns.
        pipeline: Returns the data processing pipeline.
        _preprocess: Preprocesses the input DataFrame.

    """

    concat_text_data = luigi.ListParameter()

    @property
    def feature_columns(self):
        """
        Returns the list of feature columns.

        Returns:
            list: The list of feature columns.

        """
        return ["clip_img_embeddings", "clip_text_embeddings", "clip_dot_embeddings"]

    def pipeline(self):
        """
        Returns the data processing pipeline.

        Returns:
            Pipeline: The data processing pipeline.

        """
        dino = WrappedCLIPV2(
            input_cols=["data", "text_data"],
            output_cols=[
                "clip_img_embeddings",
                "clip_text_embeddings",
                "clip_dot_embeddings",
            ],
        )
        return Pipeline(
            stages=[
                dino,
                SQLTransformer(
                    statement="SELECT ImageUniqueID, species, clip_img_embeddings, clip_text_embeddings, clip_dot_embeddings FROM __THIS__"
                ),
            ]
        )

    def _preprocess(self, df) -> DataFrame:
        """
        Preprocesses the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame.

        Returns:
            DataFrame: The preprocessed DataFrame.

        """
        text_cols = [
            F.concat(F.lit(f"{col_name} "), F.col(col_name).cast("string")).alias(col_name) for col_name in self.concat_text_data
        ]
        df = df.withColumn("text_data", F.concat_ws(", ", *text_cols))

        return df
