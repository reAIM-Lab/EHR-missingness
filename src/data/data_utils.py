import polars as pl
import shutil

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

def extract_demo(df_filtered):
    person = df_filtered.filter(pl.col("table").str.starts_with("person"))

    gender = (
        person.filter(pl.col("normalized_code").str.starts_with("Gender/"))
        .select(
            pl.coalesce([pl.col("concept_name"), pl.col("normalized_code").str.split("/").list.get(1)])
            .str.to_lowercase()
            .alias("gender")
        )
    )
    race = (
        person.filter(pl.col("normalized_code").str.starts_with("Race/"))
        .select(
            pl.coalesce([pl.col("concept_name"), pl.col("normalized_code").str.split("/").list.get(1)])
            .str.to_lowercase()
            .alias("race")
        )
    )
    ethnicity = (
        person.filter(pl.col("normalized_code").str.starts_with("Ethnicity/"))
        .select(
            pl.coalesce([pl.col("concept_name"), pl.col("normalized_code").str.split("/").list.get(1)])
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
    age = df.filter(pl.col("normalized_code") == "MEDS_BIRTH").select(pl.col("age").alias("age"))
    df_demographics = pl.concat([age, ethnicity, gender, race], how="horizontal")

    return df_demographics