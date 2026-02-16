import polars as pl
import shutil
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_mortality_csv_with_splits(config):
    """Load row-wise mortality CSV and create deterministic train/eval splits."""
    csv_path = config.get("raw_csv_path")
    if not csv_path:
        cohort = str(config.get("mortality_cohort", "all")).lower()
        cohort_csv_paths = config.get("cohort_csv_paths", {})
        if cohort_csv_paths and cohort in cohort_csv_paths:
            csv_path = cohort_csv_paths[cohort]
        else:
            default_csvs = {
                "all": f"{config['target_dir']}/raw/mortality_mimic_missingness.csv",
                "micu": f"{config['target_dir']}/raw/mortality_mimic_missingness_micu.csv",
                "ccu": f"{config['target_dir']}/raw/mortality_mimic_missingness_ccu.csv",
            }
            if cohort not in default_csvs:
                valid = sorted(default_csvs.keys())
                raise ValueError(
                    f"Unknown mortality_cohort='{cohort}'. Expected one of {valid}, "
                    "or provide raw_csv_path explicitly."
                )
            csv_path = default_csvs[cohort]

    df = pl.read_csv(csv_path)
    required_columns = {"hadm_id", "boolean_value"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(
            f"Missing required columns in mortality CSV {csv_path}: {sorted(missing_required)}"
        )

    df = df.rename({"hadm_id": "subject_id"})

    # Normalize labels to {0,1} regardless of bool/string/numeric input.
    df = df.with_columns(
        pl.col("boolean_value")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .replace({"true": "1", "false": "0"})
        .cast(pl.Int32)
        .alias("boolean_value")
    )

    # Split by unique subject/admission ids to guarantee exact sample counts.
    # Use a dedicated split seed so train/eval membership stays fixed even when
    # data_seed changes for ICL example sampling.
    subject_ids = df["subject_id"].unique().to_list()
    split_seed = int(config.get("split_seed", 0))
    rng = random.Random(split_seed)
    rng.shuffle(subject_ids)

    train_num = config.get("train_num")
    test_num = config.get("test_num")

    if train_num is not None or test_num is not None:
        if train_num is None or test_num is None:
            raise ValueError(
                "Both 'train_num' and 'test_num' must be provided together for mortality CSV splits."
            )

        train_num = int(train_num)
        test_num = int(test_num)
        if train_num < 0 or test_num < 0:
            raise ValueError("'train_num' and 'test_num' must be non-negative integers.")

        required_total = train_num + test_num
        if required_total > len(subject_ids):
            raise ValueError(
                f"Requested train_num + test_num = {required_total}, but only "
                f"{len(subject_ids)} unique subject_ids are available in {csv_path}."
            )

        train_ids = set(subject_ids[:train_num])
        eval_ids = set(subject_ids[train_num:train_num + test_num])
    else:
        # Backward compatibility for legacy configs.
        max_samples = config.get("max_samples")
        if max_samples is not None and len(subject_ids) > int(max_samples):
            subject_ids = subject_ids[: int(max_samples)]
            df = df.filter(pl.col("subject_id").is_in(subject_ids))

        train_frac = float(config.get("train_frac", 0.5))
        split_idx = int(round(train_frac * len(subject_ids)))
        train_ids = set(subject_ids[:split_idx])
        eval_ids = set(subject_ids[split_idx:])

    train_data = df.filter(pl.col("subject_id").is_in(train_ids))
    eval_data = df.filter(pl.col("subject_id").is_in(eval_ids))

    return eval_data, train_data


def generate_sepsis_cohort(n=1000):
    # --- 1. Define Multivariate Physiology (X) ---
    # Order: [WBC, MAP, Lactate, PCT]
    means = [0, 0, 0, 0] # Standardized Z-scores
    
    cov = [
        # WBC   MAP    Lact   PCT
        [ 1.0, -0.2,   0.4,   0.7],  # WBC: Strongly links to PCT (infection)
        [-0.2,  1.0,  -0.6,  -0.3],  # MAP: Strongly neg links to Lactate (shock)
        [ 0.4, -0.6,   1.0,   0.5],  # Lactate: Links to MAP & Infection
        [ 0.7, -0.3,   0.5,   1.0]   # PCT: expensive infection marker
    ]
    
    # Generate the "True" Physiology
    features = np.random.multivariate_normal(means, cov, n)
    X = pd.DataFrame(features, columns=['WBC', 'MAP', 'Lactate', 'PCT'])

    # --- 2. Define Latent Confounders ---
    
    # U: Clinical Severity
    U = 0.3 * X['Lactate'] - 0.4 * X['MAP'] + 0.4 * X['WBC'] + np.random.normal(0, 0.5, n)
    
    # W: Insurance Status / Resource Access
    # Independent of physiology (Biology doesn't care if you are rich)
    # 0 = Uninsured, 1 = Insured
    W = np.random.binomial(1, 0.8, n)

    # --- 3. Define Outcome (Y: Septic Shock) ---
    # Driven by Physiology + Severity
    logits = 2.5 * U + 0.6 * X['PCT'] + 0.3 * X['Lactate']
    Y_probs = 1 / (1 + np.exp(-logits))
    Y = np.random.binomial(1, Y_probs)

    # --- 4. Apply Missingness Mechanisms ---
    
    # Mechanism A: Clinical Intuition (Lactate)
    prob_M_Lactate = 1 / (1 + np.exp(-(U * 5 + 3))) # Doctor orders Lactate only if patient looks sick (High U)
    M_Lactate = np.random.binomial(1, prob_M_Lactate)
    
    # Mechanism B: Socioeconomic Barrier (PCT)
    # Doctor orders PCT only if Insurance covers it (W=1) OR very severe (High U)
    # Logic: "If uninsured, we can't afford this $100 test unless dying."
    prob_if_insured = 1 / (1 + np.exp(-(U * 6 - 1)))
    # prob_if_uninsured = 0.2 * (1 / (1 + np.exp(-(U * 5 - 2)))) # Logic for Uninsured (Resource Constraint)
    
    # Combine based on W
    # prob_M_PCT = W * prob_if_insured + (1 - W) * prob_if_uninsured
    prob_M_PCT = prob_if_insured
    M_PCT = np.random.binomial(1, prob_M_PCT)

    # --- 5. Final Observed Dataframe ---
    df = X.copy()
    # Mask values
    df.loc[M_Lactate == 0, 'Lactate'] = np.nan
    df.loc[M_PCT == 0, 'PCT'] = np.nan
    
    # Add metadata for your experiment (don't give this to LLM!)
    df['boolean_value'] = Y
    df['Insurance_Status'] = W
    df['True_Lactate'] = X['Lactate'] # For validation
    df['True_U'] = U
    df['True_logits'] = logits
    
    return df

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
    if config['experiment'] == 'mimic':
        df_filtered = df_filtered.filter(pl.col("code").str.starts_with("MEDS_BIRTH"))
        df = df_filtered.with_columns([
            ((pl.col("prediction_time") - pl.col("time")).dt.total_days() / 365.25)
            .round(0)
            .cast(pl.Int32)
            .alias("age")
        ])
        df_demographics = df.filter(pl.col("code") == "MEDS_BIRTH").select(pl.col("age")).unique()

    return df_demographics

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