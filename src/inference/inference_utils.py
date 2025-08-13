import polars as pl

from src.data.data_utils import extract_demo
from src.data.feature_configs import codes_to_keep

def serialize_data(data, config):

    demographics = extract_demo(data)
    markdown_str = f"\n # Electronic Health Record (Observation Window: {config['observation_window_days']} days)\n"
    markdown_str += f"## Prediction Time: {data['prediction_time'].to_list()[0]}\n"

    age_value = str(demographics['age'].to_list()[0])
    markdown_str += "## Demographics\n"
    markdown_str += "Patient age: " + age_value + "\n"
    markdown_str += "Patient gender: " + demographics['gender'].to_list()[0] + "\n"
    markdown_str += "Patient race: " + demographics['race'].to_list()[0] + "\n"

    if config['include_conditions']:
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


    if config['include_procedures']:
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
    all_measurements = codes_to_keep.keys()
    for measurement in all_measurements:
        filtered_measurements = data.filter(pl.col("code").is_in(codes_to_keep[measurement]))
        if not filtered_measurements.is_empty():
            markdown_str += f"- {measurement}\n"
            filtered_measurements = filtered_measurements.with_columns(
                pl.col("numeric_value").cast(pl.Float64).round(2).alias("numeric_value")
            )
            markdown_str += "\n".join([f"  - {row['numeric_value']} (unit: {row['unit']}) measured at {row['time']}" for row in filtered_measurements.to_dicts()]) + "\n"
        else:
            if config['explicit_missingness']:
                markdown_str += f"- {measurement}\n"
                markdown_str += "  - Not measured during observation window\n"
    

    return markdown_str

def extract_structured_data(data, config):
    demographics = extract_demo(data)
    feature_dict = {}
    age_value = demographics['age'].to_list()[0]
    feature_dict['age'] = age_value
    feature_dict['gender'] = demographics['gender'].to_list()[0]
    feature_dict['race'] = demographics['race'].to_list()[0]

    all_measurements = codes_to_keep.keys()
    for measurement in all_measurements:
        filtered_measurements = data.filter(pl.col("code").is_in(codes_to_keep[measurement]))
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

    if config['explicit_missingness']:
        instruction = f'You are a helpful assistant that can answer questions about the patient\'s electronic health record.'
        #instruction += f'Reason about the missingness in recorded measurements and answer the query.'
        instruction += f'Reason about potential reasons for the missingness in recorded measurements, and answer the query.'
        instruction += f'\nQuery: {query}'
    else:
        instruction = f'You are a helpful assistant that can answer questions about the patient\'s electronic health record.'
        instruction += f'\nQuery: {query}'

    instruction += " Return the final prediction as a percentage at the end of your response in the following format: [Final Prediction: <prediction>%]"
    return instruction

def extract_prediction(output: str) -> float:
    #Extract numeric prediction from LLM output string.
    try:
        start = output.find("[Final Prediction:") + len("[Final Prediction:")
        end = output.find("]", start)
        if start == -1 or end == -1:
            return None
        
        pred_str = output[start:end].strip().replace("%", "")
        pred = float(pred_str)
        
        if pred < 0 or pred > 100:
            return None
            
        return pred / 100.0
        
    except:
        return None