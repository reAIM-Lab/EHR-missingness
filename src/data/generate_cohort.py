from pathlib import Path
import polars as pl

from src.data.data_utils import normalize_icd10, construct_meds_dir
from src.data.feature_configs import codes_to_keep, clip_range, required_categories, codes_to_keep_mimic, required_categories_mimic

def generate_cohort_mimic(config):
    meds_dir = Path(config['meds_dir'])
    task_dir = Path(config['task_dir']) / f"{config['downstream_task']}"
    concept_path = Path(config['concept_dir'])

    concepts = pl.read_parquet(sorted(concept_path.glob("codes.parquet")))
    concepts = concepts[['code', 'description', 'parent_codes']]
    concepts = concepts.with_columns(
        concepts["parent_codes"].list.first().alias("parent_codes")
    )
    
    code_to_category = {code: category for category, codes in codes_to_keep_mimic.items() for code in codes}

    subjects = pl.read_parquet(sorted(task_dir.glob("*.parquet")))
    subjects = subjects.group_by("subject_id").agg(pl.col("prediction_time").first()).join(
        subjects, on=["subject_id", "prediction_time"], how="inner"
    )

    label_counts = subjects.group_by("boolean_value").agg(pl.count().alias("count"))
    print("Label distribution:")
    print(label_counts)

    train_path = meds_dir / "data/train"
    tune_path = meds_dir / "data/tuning"
    test_path = meds_dir / "data/held_out"

    train_files = sorted(train_path.glob("*.parquet"))
    tune_files = sorted(tune_path.glob("*.parquet"))
    test_files = sorted(test_path.glob("*.parquet"))

    unique_subject_ids = set(subjects['subject_id'].to_list())
    all_codes_to_keep = [codes for codes in codes_to_keep_mimic.values()]

    new_meds_dir = Path(config['target_dir']) / f"{config['downstream_task']}_MEDS"
    new_meds_dir.mkdir(parents=True, exist_ok=True)

    construct_meds_dir(meds_dir, new_meds_dir)

    for j, files in enumerate([train_files, tune_files, test_files]):
        for i, file in enumerate(files):
            print(f"Processing file {i+1} of {len(files)}")
            df = pl.read_parquet(file)
            df = df.filter(pl.col("subject_id").is_in(unique_subject_ids))
            df = df.join(subjects, on=["subject_id"], how="left")
            df = df.filter(pl.col("time") <= pl.col("prediction_time"))

            df = df.join(
                concepts, left_on="code", right_on="code", how="left"
            )

            filter_condition = pl.col("code").str.starts_with("LAB")
            df_measurements = df.filter(filter_condition)
            df_person = df.filter(pl.col("code").is_in(["MEDS_BIRTH"]))

            df_measurements = df_measurements.with_columns(
                pl.col("code")
                .map_elements(
                    lambda x: x.split("//")[-1] if isinstance(x, str) and ("LAB" in x) else x,
                    return_dtype=pl.Utf8
                )
                .alias("unit")
            )

            conditions = pl.col("numeric_value").is_not_null()
            df_measurements = df_measurements.filter(pl.col("numeric_value").is_not_null() & 
                (
                    (pl.col("prediction_time") - pl.col("time"))
                    .dt.total_days() <= config['observation_window_days']
                )
            )

            # top_codes = (
            #     df_measurements
            #     .filter(pl.col("parent_codes").is_not_null())
            #     .group_by(["code", "parent_codes", "description"])
            #     .agg(pl.len().alias("count"))
            #     .sort("count", descending=True)
            #     .head(100)
            # )
            # print(top_codes.select(["parent_codes", "description", "count"]).to_pandas().to_string())

            df_measurements = df_measurements.filter(
                (pl.col("parent_codes").is_in([code for codes in all_codes_to_keep for code in codes]))
            )

            # Map codes to categories
            df_measurements = df_measurements.with_columns(
                pl.col("parent_codes")
                .map_elements(
                    lambda x: code_to_category.get(x),
                    return_dtype=pl.Utf8
                )
                .alias("category")
            )
            # Keep only rows with valid categories and numeric values
            conditions = pl.col("category").is_not_null()

            if config['clip_range']:
                for cat, (low, high) in clip_range.items():
                    conditions &= pl.when(pl.col("category") == cat).then(
                        (pl.col("numeric_value") >= low) & (pl.col("numeric_value") <= high)
                    ).otherwise(True)

            # Add condition for observation window
            # conditions &= (
            #     (pl.col("prediction_time") - pl.col("time"))
            #     .dt.total_days() <= config['observation_window_days']
            # )

            # Apply the filter
            df_measurements = df_measurements.filter(conditions)

            # Sort by time descending and keep only the last measurement for each category
            df_measurements = (
                df_measurements.sort("time", descending=True)
                .group_by(["subject_id", "prediction_time", "category"], maintain_order=True)
                .head(1)
            )

            if config['require_baselines']:
                # Get subjects that have measurements for all required categories
                subject_category_counts = (
                    df_measurements.filter(pl.col("category").is_in(required_categories_mimic))
                    .group_by("subject_id")
                    .agg(pl.col("category").n_unique())
                    .filter(pl.col("category").eq(len(required_categories_mimic)))
                    .select("subject_id")
                )
                subject_ids_to_keep = subject_category_counts["subject_id"].to_list()
            else:
                subject_ids_to_keep = unique_subject_ids

            if config['require_demographics']:
                # Filter for subjects that have both MEDS_BIRTH and Gender codes
                raise ValueError(
                    "require_demographics=True is not supported. "
                    "Please set require_demographics=False to proceed."
                )

            # Filter df_visit to only include subjects with complete measurements
            df_measurements = df_measurements.filter(pl.col("subject_id").is_in(subject_ids_to_keep)).drop("category")
            df_person = df_person.filter(pl.col("subject_id").is_in(subject_ids_to_keep))
            df_meds = df_measurements.select(df_person.columns).vstack(df_person)

            df_meds = df_meds.sort(["subject_id", "time"]) # Sort each subject_id's data by time, keeping subject_id groups contiguous

            if j == 0:
                df_meds.write_parquet(new_meds_dir / "data" / "train" / file.name)
            elif j == 1:
                df_meds.write_parquet(new_meds_dir / "data" / "tuning" / file.name)
            else:
                df_meds.write_parquet(new_meds_dir / "data" / "held_out" / file.name)