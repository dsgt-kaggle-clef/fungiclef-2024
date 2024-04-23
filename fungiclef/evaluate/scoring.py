## From https://huggingface.co/picekl/FungiCLEF2024-Sample_Submission/blob/main/script.py 

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

COLUMNS = ["observationID", "class_id"]
poisonous_lvl = pd.read_csv(
    "http://ptak.felk.cvut.cz/plants//DanishFungiDataset/poison_status_list.csv"
)
POISONOUS_SPECIES = poisonous_lvl[poisonous_lvl["poisonous"] == 1].class_id.unique()


def classification_error_with_unknown(
    merged_df, cost_unkwnown_misclassified=10, cost_misclassified_as_unknown=0.1
):
    num_misclassified_unknown = sum((merged_df.class_id_gt == -1) & (merged_df.class_id_pred != -1))
    num_misclassified_as_unknown = sum(
        (merged_df.class_id_gt != -1) & (merged_df.class_id_pred == -1)
    )
    num_misclassified_other = sum(
        (merged_df.class_id_gt != merged_df.class_id_pred)
        & (merged_df.class_id_pred != -1)
        & (merged_df.class_id_gt != -1)
    )
    return (
        num_misclassified_other
        + num_misclassified_unknown * cost_unkwnown_misclassified
        + num_misclassified_as_unknown * cost_misclassified_as_unknown
    ) / len(merged_df)


def classification_error(merged_df):
    return classification_error_with_unknown(
        merged_df, cost_misclassified_as_unknown=1, cost_unkwnown_misclassified=1
    )


def num_psc_decisions(merged_df):
    # Number of observations that were misclassified as edible, when in fact they are poisonous
    num_psc = sum(
        merged_df.class_id_gt.isin(POISONOUS_SPECIES)
        & ~merged_df.class_id_pred.isin(POISONOUS_SPECIES)
    )
    return num_psc


def num_esc_decisions(merged_df):
    # Number of observations that were misclassified as poisonus, when in fact they are edible
    num_esc = sum(
        ~merged_df.class_id_gt.isin(POISONOUS_SPECIES)
        & merged_df.class_id_pred.isin(POISONOUS_SPECIES)
    )
    return num_esc


def psc_esc_cost_score(merged_df, cost_psc=100, cost_esc=1):
    return (
        cost_psc * num_psc_decisions(merged_df) + cost_esc * num_esc_decisions(merged_df)
    ) / len(merged_df)

def score_model(predicted_class: np.ndarray, gt_df: pd.DataFrame):
    
    pred_df = gt_df[["observationID"]].copy()

    try:
        pred_df['class_id'] == predicted_class
    except Exception as e:
        raise ValueError("Prediction Length Mismatch: {e}".format(e=e))
    
    gt_df = gt_df.drop_duplicates("observationID")
    pred_df = pred_df.drop_duplicates("observationID")

    if len(gt_df) != len(pred_df):
        print(f"Predictions should have {len(gt_df)} records.")
        raise ValueError(f"Predictions should have {len(gt_df)} records.")
    missing_obs = gt_df.loc[
        ~gt_df["observationID"].isin(pred_df["observationID"]),
        "observationID",
    ]

    if len(missing_obs) > 0:
        if len(missing_obs) > 3:
            missing_obs_str = ", ".join(missing_obs.iloc[:3].astype(str)) + ", ..."
        else:
            missing_obs_str = ", ".join(missing_obs.astype(str))
        print(f"Predictions is missing observations: {missing_obs_str}")
        raise ValueError(f"Predictions is missing observations: {missing_obs_str}")

    # merge dataframes
    merged_df = pd.merge(
        gt_df,
        pred_df,
        how="outer",
        on="observationID",
        validate="one_to_one",
        suffixes=("_gt", "_pred"),
    )

    # evaluate accuracy_score and f1_score
    cls_error = classification_error(merged_df)
    cls_error_with_unknown = classification_error_with_unknown(merged_df)
    psc_esc_cost = psc_esc_cost_score(merged_df)

    result = [
        {
            "test_split": {
                "F1 Score": np.round(
                    f1_score(merged_df["class_id_gt"], merged_df["class_id_pred"], average="macro")
                    * 100,
                    2,
                ),
                "Track 1: Classification Error": np.round(cls_error, 4),
                "Track 2: Cost for Poisonousness Confusion": np.round(psc_esc_cost, 4),
                "Track 3: User-Focused Loss": np.round(cls_error + psc_esc_cost, 4),
                "Track 4: Classification Error with Special Cost for Unknown": np.round(
                    cls_error_with_unknown, 4
                ),
            }
        }
    ]

    print(f"Evaluated scores: {result[0]['test_split']}")

    return result