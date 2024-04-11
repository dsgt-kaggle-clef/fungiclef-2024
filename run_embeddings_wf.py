import argparse
from fungiclef.workflows.embeddings_wf import run_embeddings_wf

parser = argparse.ArgumentParser(description="Luigi pipeline")
parser.add_argument(
    "--gcs-root-path",
    type=str,
    default="gs://dsgt-clef-fungiclef-2024",
    help="Root directory for plantclef-2024 in GCS",
)
parser.add_argument(
    "--train-data-path",
    type=str,
    default="data/parquet/DF20",
    help="Root directory for training data in GCS",
)
parser.add_argument(
    "--output-name-path",
    type=str,
    default="data/parquet/DF20_embeddings",
    help="GCS path for output Parquet files",
)


args = parser.parse_args()

input_path = f"{args.gcs_root_path}/{args.train_data_path}"
output_path = f"{args.gcs_root_path}/{args.output_name_path}"

run_embeddings_wf(input_path=input_path, output_path=output_path)