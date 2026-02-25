import polars as pl
import random
import pandas as pd


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

def sample_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    current_subjects = df['subject_id'].unique().to_list()
    if len(current_subjects) > config['max_samples']:
        rng = random.Random(config["data_seed"])
        sampled_subjects = rng.sample(current_subjects, config['max_samples'])
        df = df.filter(pl.col("subject_id").is_in(sampled_subjects))
        print(f"Further sampled to {config['max_samples']} subjects")
    
    return df
