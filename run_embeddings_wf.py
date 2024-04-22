import argparse
from fungiclef.workflows.embeddings_wf import ClipEmbedWorkflow, DinoDctnEmbedWorkflow
import luigi
from fungiclef.utils import read_config

config = read_config("fungiclef/config.json")

parser = argparse.ArgumentParser(description="Luigi pipeline")
parser.add_argument(
    "--input-data-path",
    type=str,
    default=f"{config['gs_paths']['train_and_test_300px_w_test_meta']['train_parquet']}",
    help="Root directory for training data in GCS",
)
parser.add_argument(
    "--output-name-path",
    type=str,
    default=f"{config['gs_paths']['train_and_test_300px_w_test_meta']['train_embedding_dir']}",
    help="GCS path for output Parquet files",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=10,
    help="Number of samples to run the pipeline on",
)
parser.add_argument(
    "--workflow",
    type=str,
    default="dino_dctn",
    help="Task to run the pipeline on",
)
args = parser.parse_args()


input_path = args.input_data_path
output_path = args.output_name_path
num_samples = args.num_samples
workflow = args.workflow


def run():
    if workflow not in ["dino_dctn", "clip"]:
        raise ValueError("Invalid workflow")

    luigi.build(
        [
            DinoDctnEmbedWorkflow(input_path=input_path, output_path=output_path, num_samples=num_samples)
            if workflow == "dino_dctn"
            else ClipEmbedWorkflow(input_path=input_path, output_path=output_path, num_samples=num_samples)
            if workflow == "clip"
            else None
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )


run()
