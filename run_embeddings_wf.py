import argparse
from fungiclef.workflows.embeddings_wf import run_embeddings_wf

parser = argparse.ArgumentParser(description="Luigi pipeline")
parser.add_argument(
    "--gcs-root-path",
    type=str,
    default="gs://dsgt-clef-fungiclef-2024",
    help="Root directory for fungiclef-2024 in GCS",
)
parser.add_argument(
    "--train-data-path",
    type=str,
    default="dev/dev_train",
    #default="data/parquet/DF20_300px_and_DF21_300px_train",
    help="Root directory for training data in GCS",
)
parser.add_argument(
    "--output-name-path",
    type=str,
    default="dev/dev_embedded",
    #default="data/parquet/DF20_300px_and_DF21_300px_train_embedding",
    help="GCS path for output Parquet files",
)


args = parser.parse_args()

input_path = f"{args.gcs_root_path}/{args.train_data_path}"
output_path = f"{args.gcs_root_path}/{args.output_name_path}"

run_embeddings_wf(input_path=input_path, output_path=output_path, num_samples=0)