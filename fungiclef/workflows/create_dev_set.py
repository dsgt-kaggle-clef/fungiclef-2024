import pyspark.sql.functions as f

from fungiclef.utils import get_spark, read_config
from fungiclef.workflows.train_test_split import train_test_split, manipulate_class_id


def main():
    """
    Main function that processes data and writes the output dataframe to GCS.

    This function reads the configuration file, initializes Spark, loads the input DataFrame from a Parquet file,
    filters the DataFrame to create a subset based on specified families and species, calculates statistics on the dataset,
    manipulates the class_id column, splits the dataset into train and test DataFrames, and writes the output DataFrames to GCS.
    """

    # Read the configuration file
    config = read_config("fungiclef/config.json")

    # Initialize Spark
    spark = get_spark()

    # Path to the images and metadata in a dataframe of train and test 300px
    df_path = config["gs_paths"]["train_and_test_300px_w_test_meta"]["raw_parquet"]

    # Output_paths
    dev_train_output_path = config["gs_paths"]["dev_set"]["train"]
    dev_test_output_path = config["gs_paths"]["dev_set"]["test"]

    # Load the DataFrame from the Parquet file
    df = spark.read.parquet(df_path)

    # subset df to create dev set
    # Only take fungi of following three families
    considered_families = ["Russulaceae", "Boletaceae", "Amanitaceae"]

    family_subset = df.filter(df.family.isin(considered_families))

    # only take fungi of the following species
    selected_mushrooms = [
        "Neoboletus luridiformis (Rostk.) Gelardi, Simonini & Vizzini, 2014",
        "Imleria badia (Fr.) Vizzini, 2014",
        "Amanita muscaria (L.) Lam., 1783",
        "Russula ochroleuca (Pers.) Fr.",
        "Russula nigricans (Bull.) Fr.",
        "Lactarius blennius (Fr.) Fr.",
    ]

    species_subset = family_subset.filter(family_subset.scientificName.isin(selected_mushrooms))

    # validation
    total_count = df.count()
    dev_subset_count = species_subset.count()
    print("DataFrame Length:")
    print(f"Total Dataset: {total_count}")
    print(f"Dev Subset: {dev_subset_count}")
    print(f"Dev Fraction: {dev_subset_count/total_count}")

    # posionus in total and subset
    sum_poisonous = df.select(f.sum("poisonous")).collect()[0][0]
    percentage_poisonous = sum_poisonous / total_count
    print("Poisnous statistics:")
    print(f"Total Dataset has {sum_poisonous} poisonous mushrooms")
    print(f"Total Dataset has {percentage_poisonous} poisonous mushrooms")

    sum_poisonous_sub = species_subset.select(f.sum("poisonous")).collect()[0][0]
    percentage_poisonous_sub = sum_poisonous_sub / dev_subset_count
    print(f"Dev Subset has {sum_poisonous_sub} poisonous mushrooms")
    print(f"Dev Subset has {percentage_poisonous_sub} poisonous mushrooms")

    # Prep class_id column
    # TODO: does not find any unknwon or unlabled class_ids given the filters before
    species_subset = manipulate_class_id(species_subset)

    # Create image dataframe
    dev_train_df, dev_test_df = train_test_split(
        df=species_subset,
        train_pct=0.8,
        stratify_col="class_id",
    )

    # Write the DataFrame to GCS in Parquet format
    dev_train_df.repartition(1).write.mode("overwrite").parquet(dev_train_output_path)
    dev_test_df.repartition(1).write.mode("overwrite").parquet(dev_test_output_path)


if __name__ == "__main__":
    main()
