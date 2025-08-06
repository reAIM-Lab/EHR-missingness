from pathlib import Path
import polars as pl

from src.data.data_utils import normalize_icd10, construct_meds_dir
from src.data.feature_configs import codes_to_keep, clip_range, required_categories

def generate_cohort(config):

    meds_dir = Path(config['meds_dir'])
    task_dir = Path(config['task_dir']) / f"{config["downstream_task"]}"
    concept_path = Path(config['concept_dir'])

    train_path = meds_dir / "data/train"
    tune_path = meds_dir / "data/tuning"
    test_path = meds_dir / "data/held_out"

    # Load concept table from OMOP to retrieve concept names and domain_ids
    concepts = pl.read_parquet(sorted(concept_path.glob("*.parquet")))
    concepts = concepts.with_columns(
        (pl.col("vocabulary_id").cast(pl.Utf8) + "/" + pl.col("concept_code").cast(pl.Utf8)).alias("vocabulary_concept")
    )
    # print(concepts)
    # print(concepts.columns)
    # print("Unique domain_ids:", concepts["domain_id"].unique().to_list())

    concepts = concepts[['concept_name', 'vocabulary_concept', 'domain_id']]
    concepts = normalize_icd10(concepts, 'vocabulary_concept')
    code_to_category = {code: category for category, codes in codes_to_keep.items() for code in codes}

    subjects = pl.read_parquet(sorted(task_dir.glob("*.parquet")))
    subjects = subjects.group_by("subject_id").agg(pl.col("prediction_time").first()).join(
        subjects, on=["subject_id", "prediction_time"], how="inner"
    )

    print(len(subjects))

    train_files = sorted(train_path.glob("*.parquet"))
    tune_files = sorted(tune_path.glob("*.parquet"))
    test_files = sorted(test_path.glob("*.parquet"))

    unique_subject_ids = set(subjects['subject_id'].to_list())
    all_codes_to_keep = [codes for codes in codes_to_keep.values()]

    new_meds_dir = Path(config['target_dir']) / f"{config['downstream_task']}_MEDS"
    new_meds_dir.mkdir(parents=True, exist_ok=True)

    construct_meds_dir(meds_dir, new_meds_dir)

    for j, files in enumerate([train_files, tune_files, test_files]):
    #for j, files in enumerate([train_files]):
        for i, file in enumerate(files):
            print(f"Processing file {i+1} of {len(files)}")
            df = pl.read_parquet(file)
            df = df.filter(pl.col("subject_id").is_in(unique_subject_ids))
            df = df.join(subjects, on=["subject_id"], how="left")
            df = df.filter(pl.col("time") <= pl.col("prediction_time"))

            df = df.with_columns(
                pl.col("code")
                .map_elements(
                    lambda x: x.split("//")[0] if isinstance(x, str) and ("LOINC" in x or "SNOMED" in x) else x,
                    return_dtype=pl.Utf8
                )
                .alias("code")
            )

            df = normalize_icd10(df, 'code')
            df = df.join(
                concepts, left_on="normalized_code", right_on="normalized_vocabulary_concept", how="left"
            )
            
            filter_condition = pl.col("table").str.starts_with("measurement")
            df_measurements = df.filter(filter_condition)
            df_person = df.filter(pl.col("table").str.starts_with("person"))

            df_measurements = df_measurements.filter(
                (pl.col("code").is_in([code for codes in all_codes_to_keep for code in codes]))
            )

            # Map codes to categories
            df_measurements = df_measurements.with_columns(
                pl.col("code")
                .map_elements(
                    lambda x: code_to_category.get(x),
                    return_dtype=pl.Utf8
                )
                .alias("category")
            )
            # Keep only rows with valid categories and numeric values
            conditions = pl.col("category").is_not_null() & pl.col("numeric_value").is_not_null()

            if config['clip_range']:
                for cat, (low, high) in clip_range.items():
                    conditions &= pl.when(pl.col("category") == cat).then(
                        (pl.col("numeric_value") >= low) & (pl.col("numeric_value") <= high)
                    ).otherwise(True)

            # Add condition for observation window
            conditions &= (
                (pl.col("prediction_time") - pl.col("time"))
                .dt.total_days() <= config['observation_window_days']
            )

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
                    df_measurements.filter(pl.col("category").is_in(required_categories))
                    .group_by("subject_id")
                    .agg(pl.col("category").n_unique())
                    .filter(pl.col("category").eq(len(required_categories)))
                    .select("subject_id")
                )
            
            if config['require_demographics']:
                # Filter for subjects that have both MEDS_BIRTH and Gender codes
                demo_requirements = (
                    df_person
                    .filter(
                        pl.col("code").is_in(["MEDS_BIRTH"]) |
                        pl.col("code").str.starts_with("Gender/") |
                        pl.col("code").str.starts_with("Race/") |
                        pl.col("code").str.starts_with("Ethnicity/")
                    )
                    .group_by("subject_id")
                    .agg([
                        pl.col("code").eq("MEDS_BIRTH").any().alias("has_birth"),
                        pl.col("code").str.starts_with("Gender/").any().alias("has_gender"),
                        (
                            pl.col("code").str.starts_with("Race/") &
                            ~pl.col("code").str.to_lowercase().str.contains("unknown")
                        ).any().alias("has_valid_race"),
                        (
                            pl.col("code").str.starts_with("Ethnicity/") &
                            ~pl.col("code").str.to_lowercase().str.contains("unknown")
                        ).any().alias("has_valid_ethnicity"),
                    ])
                    .filter(
                        pl.col("has_birth") &
                        pl.col("has_gender") &
                        pl.col("has_valid_race") &
                        pl.col("has_valid_ethnicity")
                    )
                    .select("subject_id")
                )

            if config['require_baselines'] and config['require_demographics']:
                subject_ids_to_keep = subject_category_counts.join(demo_requirements, on="subject_id", how="inner")["subject_id"].to_list()
            elif config['require_baselines'] or config['require_demographics']:
                if config['require_baselines']:
                    subject_ids_to_keep = subject_category_counts["subject_id"].to_list()
                else:
                    subject_ids_to_keep = demo_requirements["subject_id"].to_list()

            # Filter df_visit to only include subjects with complete measurements
            df_measurements = df_measurements.filter(pl.col("subject_id").is_in(subject_ids_to_keep)).drop("category")
            df_person = df_person.filter(pl.col("subject_id").is_in(subject_ids_to_keep))
            df_meds = df_measurements.select(df_person.columns).vstack(df_person)

            if config['include_conditions']:
                df_conditions = df.filter(
                    (pl.col("table").str.starts_with("condition")) &
                    (pl.col("subject_id").is_in(subject_ids_to_keep)) &
                    ((pl.col("prediction_time") - pl.col("time")).dt.total_days() <= config['observation_window_days'])
                )

                df_meds = df_meds.vstack(df_conditions)
            if config['include_procedures']:
                df_procedures = df.filter(
                    (pl.col("table").str.starts_with("procedure")) &
                    (pl.col("subject_id").is_in(subject_ids_to_keep)) &
                    ((pl.col("prediction_time") - pl.col("time")).dt.total_days() <= config['observation_window_days'])
                )
                df_meds = df_meds.vstack(df_procedures)

            df_meds = df_meds.sort(["subject_id", "time"]) # Sort each subject_id's data by time, keeping subject_id groups contiguous
            
            if j == 0:
                df_meds.write_parquet(new_meds_dir / "data" / "train" / file.name)
            elif j == 1:
                df_meds.write_parquet(new_meds_dir / "data" / "tuning" / file.name)
            else:
                df_meds.write_parquet(new_meds_dir / "data" / "held_out" / file.name)
