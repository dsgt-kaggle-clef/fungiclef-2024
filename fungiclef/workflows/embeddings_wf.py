import luigi
import luigi.contrib.gcs
from fungiclef.workflows.embeddings_tasks import ProcessDino, ProcessCLIP, ProcessDCTN


class DinoDctnEmbedWorkflow(luigi.Task):
    """
    A Luigi workflow for embedding Dino and DCTN data.

    Parameters:
    - input_path (str): The path to the input data.
    - output_path (str): The path to save the output data.
    - num_samples (int): The number of samples to process.

    Usage:
    - Instantiate the `DinoDctnEmbedWorkflow` class with the required parameters.
    - Call the `run` method to execute the workflow.
    """

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_samples = luigi.Parameter()

    def run(self):
        # Run jobs with subset and full-size data
        for subset in [True, False]:
            final_output_path = self.output_path
            if subset:
                subset_path = f"subset_{self.output_path.split('/')[-1]}"
                final_output_path = self.output_path.replace(self.output_path.split("/")[-1], subset_path)

            # Splitting the data into batches
            if self.num_samples > 0:
                yield [
                    ProcessDino(
                        input_path=self.input_path,
                        output_path=f"{final_output_path}/dino",
                        should_subset=subset,
                        sample_id=i,
                        num_sample_id=self.num_samples,
                    )
                    for i in range(self.num_samples)
                ]

            # Alternatively, if you want to run the pipeline on the full dataset
            else:
                yield ProcessDino(
                    input_path=self.input_path,
                    output_path=f"{final_output_path}/dino",
                    should_subset=subset,
                    sample_id=None,
                    num_sample_id=self.num_samples,
                )

            # Batching not necessary for DCT
            yield [
                ProcessDCTN(
                    input_path=f"{final_output_path}/dino/data",
                    output_path=f"{final_output_path}/dino_dct",
                    should_subset=subset,
                    filter_size=8,
                ),
                ProcessDCTN(
                    input_path=f"{final_output_path}/dino/data",
                    output_path=f"{final_output_path}/dino_dct_16",
                    should_subset=subset,
                    filter_size=16,
                ),
            ]


class ClipEmbedWorkflow(luigi.Task):
    """
    A Luigi workflow task for processing CLIP embeddings.

    Parameters:
    - input_path (str): The path to the input data.
    - output_path (str): The path to save the output data.
    - num_samples (int): The number of samples to process.

    Usage:
    - Instantiate the `ClipEmbedWorkflow` class with the required parameters.
    - Call the `run` method to execute the workflow.
    """

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_samples = luigi.Parameter()

    def run(self):
        # Run jobs with subset and full-size data
        for subset in [True, False]:
            final_output_path = self.output_path
            if subset:
                subset_path = f"subset_{self.output_path.split('/')[-1]}"
                final_output_path = self.output_path.replace(self.output_path.split("/")[-1], subset_path)

            # Splitting the data into batches
            if self.num_samples > 0:
                yield [
                    ProcessCLIP(
                        input_path=self.input_path,
                        output_path=f"{final_output_path}/clip",
                        should_subset=subset,
                        sample_id=i,
                        num_sample_id=self.num_samples,
                        concat_text_data=[
                            "locality",
                            "level0Gid",
                            "level1Gid",
                            "level2Gid",
                            "Substrate",
                            "Habitat",
                            "MetaSubstrate",
                        ],
                    )
                    for i in range(self.num_samples)
                ]

            # Alternatively, if you want to run the pipeline on the full dataset
            else:
                yield ProcessCLIP(
                    input_path=self.input_path,
                    output_path=f"{final_output_path}/clip",
                    should_subset=subset,
                    sample_id=None,
                    num_sample_id=self.num_samples,
                    concat_text_data=[
                        "locality",
                        "level0Gid",
                        "level1Gid",
                        "level2Gid",
                        "Substrate",
                        "Habitat",
                        "MetaSubstrate",
                    ],
                )
