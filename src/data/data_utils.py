import polars as pl
import shutil
import random

import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_icd10(df: pl.DataFrame, column: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(column).is_not_null() & pl.col(column).str.starts_with("ICD10CM/"))
        .then(
            ("ICD10CM/" + pl.col(column)
             .str.replace("ICD10CM/", "", literal=True)
             .str.replace(r"\.?0+$", "", literal=False))
        )
        .otherwise(pl.col(column))
        .alias(f"normalized_{column}")
    )

def construct_meds_dir(old_meds_dir, new_meds_dir):
    # Copy metadata directory from old_meds_dir to new_meds_dir
    metadata_src = old_meds_dir / "metadata"
    metadata_dst = new_meds_dir / "metadata"
    if metadata_src.exists():
        if metadata_dst.exists():
            shutil.rmtree(metadata_dst)
        shutil.copytree(metadata_src, metadata_dst)

    for split in ["train", "tuning", "held_out"]:
        data_dir = new_meds_dir / "data" / split
        data_dir.mkdir(parents=True, exist_ok=True)

def extract_demo(df_filtered, config):
    if config['experiment'] == 'cuimc': 
        person = df_filtered.filter(pl.col("table").str.starts_with("person"))
    

        gender = (
            person.filter(pl.col("code").str.starts_with("Gender/"))
            .select(
                pl.coalesce([pl.col("concept_name"), pl.col("code").str.split("/").list.get(1)])
                .str.to_lowercase()
                .alias("gender")
            )
        )
        race = (
            person.filter(pl.col("code").str.starts_with("Race/"))
            .select(
                pl.coalesce([pl.col("concept_name"), pl.col("code").str.split("/").list.get(1)])
                .str.to_lowercase()
                .alias("race")
            )
        )
        ethnicity = (
            person.filter(pl.col("code").str.starts_with("Ethnicity/"))
            .select(
                pl.coalesce([pl.col("concept_name"), pl.col("code").str.split("/").list.get(1)])
                .str.to_lowercase()
                .alias("ethnicity")
            )
        )
        df = df_filtered.with_columns([
            ((pl.col("prediction_time") - pl.col("time")).dt.total_days() / 365.25)
            .round(0)
            .cast(pl.Int32)
            .alias("age")
        ])
        age = df.filter(pl.col("code") == "MEDS_BIRTH").select(pl.col("age").alias("age"))
        df_demographics = pl.concat([age, ethnicity, gender, race], how="horizontal")
    
    elif config['experiment'] == 'mimic':
        df_filtered = df_filtered.filter(pl.col("code").str.starts_with("MEDS_BIRTH"))
        df = df_filtered.with_columns([
            ((pl.col("prediction_time") - pl.col("time")).dt.total_days() / 365.25)
            .round(0)
            .cast(pl.Int32)
            .alias("age")
        ])
        df_demographics = df.filter(pl.col("code") == "MEDS_BIRTH").select(pl.col("age")).unique()

    return df_demographics

def balance_dataset(df: pl.DataFrame, config: dict, return_prevalence: bool) -> pl.DataFrame:
    """
    Balances the dataset by sampling negative cases to match positive cases.
    """
    subject_groups = df.partition_by("subject_id", as_dict=True)

    # Get label per each subject_id
    subject_labels = {}
    for subject_id, subject_data in subject_groups.items():
        label = subject_data["boolean_value"][0]
        subject_labels[subject_id[0]] = label
    
    positive_subjects = [sid for sid, label in subject_labels.items() if label == 1]
    negative_subjects = [sid for sid, label in subject_labels.items() if label == 0]

    # Sample negative subjects to match positive subjects count
    num_positive = len(positive_subjects)
    
    if len(negative_subjects) > num_positive:
        # Use data_seed for reproducible sampling
        rng = random.Random(config["data_seed"])
        sampled_negative_subjects = rng.sample(negative_subjects, num_positive)
    else:
        sampled_negative_subjects = negative_subjects
    
    # Combine positive and sampled negative subjects
    balanced_subject_ids = positive_subjects + sampled_negative_subjects
    
    # Filter the original dataframe to only include balanced subjects
    balanced_df = df.filter(pl.col("subject_id").is_in(balanced_subject_ids))
    
    print(f"Original dataset: {len(positive_subjects)} positive, {len(negative_subjects)} negative")
    print(f"Balanced dataset: {len(positive_subjects)} positive, {len(sampled_negative_subjects)} negative")

    # Further sample the balanced dataset if max_samples is specified
    if 'max_samples' in config and config['max_samples'] is not None:
        current_subjects = balanced_df['subject_id'].unique().to_list()
        if len(current_subjects) > config['max_samples']:
            rng = random.Random(config["data_seed"])
            sampled_subjects = rng.sample(current_subjects, config['max_samples'])
            balanced_df = balanced_df.filter(pl.col("subject_id").is_in(sampled_subjects))
            print(f"Further sampled to {config['max_samples']} subjects")
    
    if return_prevalence:
        return balanced_df, len(positive_subjects)/len(negative_subjects)

    return balanced_df

def sample_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    current_subjects = df['subject_id'].unique().to_list()
    if len(current_subjects) > config['max_samples']:
        rng = random.Random(config["data_seed"])
        sampled_subjects = rng.sample(current_subjects, config['max_samples'])
        df = df.filter(pl.col("subject_id").is_in(sampled_subjects))
        print(f"Further sampled to {config['max_samples']} subjects")
    
    return df


def preprocess_df(df : pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes continuous columns and one-hot encodes categorical columns.
    Returns a new DataFrame.
    """
    df = df.copy()
    # Replace "unknown" values with NaN
    df = df.replace("unknown", float('nan'))

    exclude_cols = {'label', 'boolean_value', 'y', 'subject_id', 'split'}
    # Convert columns with only 2 unique non-nan values to binary 0/1
    for col in df.columns:
        if col in exclude_cols:
            continue 
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            # Map the two values to 0 and 1
            val_map = {val: i for i, val in enumerate(unique_vals)}
            df[col] = df[col].map(val_map)

    # Identify continuous and categorical columns
    continuous_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # Remove label columns if present
    for col in exclude_cols:
        if col in continuous_cols:
            continuous_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
    # Standardize continuous columns
    if continuous_cols:
        scaler = StandardScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    # One-hot encode categorical columns
    if categorical_cols:
        for col in categorical_cols:
            # Check if binary categorical
            df = pd.get_dummies(df, columns=[col], drop_first=False)
    return df    