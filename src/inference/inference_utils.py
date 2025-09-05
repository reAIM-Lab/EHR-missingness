import re
import polars as pl

from src.data.data_utils import extract_demo
from src.data.feature_configs import codes_to_keep, codes_to_keep_mimic

def serialize_data(data, config):

    demographics = extract_demo(data, config)
    markdown_str = f"\n # Electronic Health Record (Observation Window: {config['observation_window_days']} days)\n"
    markdown_str += f"## Prediction Time: {data['prediction_time'].cast(pl.Date).to_list()[0]}\n"

    age_value = str(demographics['age'].to_list()[0])
    markdown_str += "## Demographics\n"
    markdown_str += "Patient age: " + age_value + "\n"

    if 'gender' in demographics.columns:
        markdown_str += "Patient gender: " + demographics['gender'].to_list()[0] + "\n"

    if 'race' in demographics.columns:
        markdown_str += "Patient race: " + demographics['race'].to_list()[0] + "\n"

    if not config['labs_only']:
        markdown_str += "## Conditions\n"
        filtered_conditions = (
            data
            .filter(pl.col("table").str.starts_with("condition"))
            .filter(pl.col("concept_name").is_not_null())
            .sort("time", descending=True)  # Ensure sorting by time
        )
        if not filtered_conditions.is_empty():
            conditions = filtered_conditions["concept_name"].unique().to_list()[:config['num_conditions']]  # Extract condition names and limit
            markdown_str += "\n".join([f"- {cond}" for cond in conditions]) + "\n"

        markdown_str += "## Procedures\n"
        filtered_proc = (
            data
            .filter(pl.col("table").str.starts_with("procedure"))
            .filter(pl.col("concept_name").is_not_null())
            .sort("time", descending=True)
        )

        if not filtered_proc.is_empty():
            procedures = filtered_proc["concept_name"].unique().to_list()[:config['num_procedures']]  # Extract procedure names and limit
            markdown_str += "\n".join([f"- {cond}" for cond in procedures]) + "\n"

    markdown_str += "## Most Recent Measurements\n"

    if config['experiment'] == 'mimic':
        all_measurements = codes_to_keep_mimic.keys()

    for measurement in all_measurements:
        if config['experiment'] == 'mimic':
            filtered_measurements = data.filter(pl.col("parent_codes").is_in(codes_to_keep_mimic[measurement]))
        if not filtered_measurements.is_empty():
            markdown_str += f"- {measurement}\n"
            filtered_measurements = filtered_measurements.with_columns(
                [
                    pl.col("numeric_value").cast(pl.Float64).round(2).alias("numeric_value"),
                    pl.col("time").cast(pl.Date).alias("date")  # keep only YYYY-MM-DD
                ]
            )
            markdown_str += "\n".join([f"  - {row['numeric_value']} (unit: {row['unit']}) measured at {row['date']}" for row in filtered_measurements.to_dicts()]) + "\n"
        else:
            if config['explicit_missingness']:
                markdown_str += f"- {measurement}\n"
                markdown_str += "  - Not measured during observation window\n"
    
    return markdown_str

def extract_structured_data(data, config):
    demographics = extract_demo(data, config)
    feature_dict = {}
    age_value = demographics['age'].to_list()[0]
    feature_dict['age'] = age_value

    if config['experiment'] == 'mimic':
        all_measurements = codes_to_keep_mimic.keys()
        for measurement in all_measurements:
            filtered_measurements = data.filter(pl.col("parent_codes").is_in(codes_to_keep_mimic[measurement]))
            if not filtered_measurements.is_empty():
                filtered_measurements = filtered_measurements.with_columns(
                    pl.col("numeric_value").cast(pl.Float64).alias("numeric_value")
                )
                feature_dict[measurement] = filtered_measurements["numeric_value"].to_list()[0]
            else:
                feature_dict[measurement] = None

    return feature_dict


def get_detailed_instruct(config) -> str:
    query = config['task_query']

    instruction = f'You are a helpful assistant that can answer questions about the patient\'s electronic health record.'
    if config['include_missingness_prompt']:
        instruction += (
            " In addition to the observed values, consider the missingness of recorded measurements "
            "as potentially informative. Explicitly reason about why certain measurements might be missing, "
            "and how their absence (not being measured) could affect your answer."
        )
        instruction += f'\nQuery: {query}'
    else:
        instruction += f'\nQuery: {query}'

    instruction += " Provide the final prediction as a percentage at the end of your response in the following format: [Final Prediction: <prediction>%]"
    return instruction

def extract_prediction(output: str) -> float:
    """
    Extract numeric prediction (0-1 float) from LLM output string,
    only if the prediction is explicitly given as a percentage.
    """
    try:
        # Require % sign
        match = re.search(r"Final Prediction:\s*([\d.]+)%", output, re.IGNORECASE)
        if not match:
            return None

        pred_str = match.group(1)
        pred = float(pred_str) / 100.0
        return pred
    except Exception:
        return None