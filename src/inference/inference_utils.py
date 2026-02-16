import re
import json
import math
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


def serialize_sim_data(data, config):
    def _get_first_value(frame, column):
        if isinstance(frame, pl.DataFrame):
            if column not in frame.columns:
                return None
            return frame[column][0]
        if not hasattr(frame, "columns") or column not in frame.columns:
            return None
        return frame[column].iloc[0]

    def _is_missing(value):
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        return False

    markdown_str = "\n # Electronic Health Record\n"
    markdown_str += "## Measurements\n"
    for measurement in ["WBC", "MAP", "Lactate", "PCT"]:
        value = _get_first_value(data, measurement)
        if not _is_missing(value):
            try:
                value_str = f"{float(value):.2f}"
            except Exception:
                value_str = str(value)
            markdown_str += f"- {measurement}\n"
            markdown_str += f"  - {value_str} (standardized; z-score)\n"
        else:
            if config.get("explicit_missingness"):
                markdown_str += f"- {measurement}\n"
                markdown_str += "  - Not measured\n"

    return markdown_str

def serialize_mortality_csv_data(data, config):
    def _get_first_value(frame, column):
        if isinstance(frame, pl.DataFrame):
            if column not in frame.columns:
                return None
            return frame[column][0]
        return None

    def _is_missing(value):
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        return False

    markdown_str = "\n # Electronic Health Record\n"
    markdown_str += "## Demographics\n"
    age = _get_first_value(data, "Age")
    if not _is_missing(age):
        markdown_str += f"Patient age: {float(age):.1f}\n"

    gender = _get_first_value(data, "Gender")
    if not _is_missing(gender):
        markdown_str += f"Patient gender: {str(gender)}\n"

    markdown_str += "## Most Recent Measurements\n"
    excluded = {
        "subject_id",
        "hadm_id",
        "boolean_value",
        "split",
        "Age",
        "Gender",
        "ICU Type",
        "Operation Type",
        "Ventilation Status",
        "suspected_infection_time",
        "prediction_time",
    }
    def _normalize_col_name(name):
        return str(name).strip().lower().replace("_", " ")

    excluded_normalized = {_normalize_col_name(col) for col in excluded}
    candidate_measurements = [
        col for col in data.columns if _normalize_col_name(col) not in excluded_normalized
    ]

    for measurement in candidate_measurements:
        value = _get_first_value(data, measurement)
        if not _is_missing(value):
            try:
                value_str = f"{float(value):.2f}"
            except Exception:
                value_str = str(value)
            markdown_str += f"- {measurement}\n"
            markdown_str += f"  - {value_str}\n"
        elif config.get("explicit_missingness"):
            markdown_str += f"- {measurement}\n"
            markdown_str += "  - Not measured\n"

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
    if config['include_cot_prompt']:
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

# def get_detailed_instruct_sim(config) -> str:
#     query = config['task_query']

#     instruction = (
#         "You are a helpful assistant that can answer questions about a simulated patient record. "
#     )
#     instruction += f'Task: {query}\n'
    
#     instruction += "You must output a valid JSON object. Do not add markdown or conversational text.\n"

#     instruction += "Output Schema:\n{"
#     instruction += '\n  "clinical_reasoning": "Analyze risk based on observed physiology using clinical knowledge",'

#     if config['include_cot_prompt']:
#         instruction += '\n  "missing_data_reasoning": "Analyze WHY features are measured or missing and consider whether their absence is potentially informative",'
#         instruction += '\n  "physiological_imputation": "If any feature is missing, estimate the likely values of missing features based on the observed ones (e.g., \'Low MAP suggests Lactate is likely high\').",'
    
#     # 3. ICL Pattern Recognition (The Steering Block)
#     if config['icl_num_examples'] > 0:
#         if config['include_cot_prompt']:
#             # Explicitly points to Missingness -> Outcome link
#             instruction += '\n  "pattern_recognition": "Analyze hospital-specific patterns inferred from the labeled examples (e.g., empirical correlations between observed values or missingness and outcome)",'
#         else:
#             # Blinded: Points only to Value -> Outcome link
#             instruction += '\n  "pattern_recognition": "Analyze hospital-specific patterns inferred from the labeled examples (e.g., empirical correlations between observed values and outcome)",'
    
#     instruction += '\n  "prediction_prob": 0.0 to 1.0 (Float representing risk of Septic Shock)'
#     instruction += '\n}'

#     return instruction

def get_detailed_instruct_sim(config) -> str:
    # 1. Base Persona & Task
    instruction = (
        "You are an expert Clinical Risk Estimation System analyzing a synthetic patient record. "
        "Your goal is to estimate the risk of Septic Shock within 24 hours.\n"
    )
    
    # 2. Structure for Free-Form Reasoning
    instruction += "\nPlease provide your analysis step-by-step using the following structure:\n"
    
    # Section A: Clinical Baseline
    instruction += "\n1. CLINICAL ASSESSMENT: Analyze the patient's risk based on the observed physiology (WBC, MAP, etc.)."

    # Section B: The Steering Block (Dynamic)
    if config["include_cot_prompt"]:
        instruction += (
            "\n2. MISSINGNESS MECHANISM: Analyze WHY specific features are missing. "
            "Consider whether their absence is potentially informative of the outcome."
        )
        # instruction += (
        #     "\n3. IMPUTATION: Estimate plausible values or clinical severity signals for "
        #     "missing features using the observed ones."
        # )

    # Section C: ICL Pattern Recognition
    if config["icl_num_examples"] > 0:
        if config["include_cot_prompt"]:
            instruction += (
                "\n3. PATTERN RECOGNITION: Look at the few-shot examples provided. "
                "Identify any hospital-specific risk patterns and correlations using (a) observed values and (b) whether a feature is measured or not)."
            )
        else:
            instruction += (
                "\n2. PATTERN RECOGNITION: Look at the few-shot examples provided. "
                "Identify any hospital-specific risk patterns and correlations using observed values."
            )
    # 3. Final Output Constraint (The JSON Anchor)
    instruction += (
        "\n\nAfter your analysis, you must output the final probability in a strictly valid JSON block "
        "at the very end of your response. Use this format:\n"
        "```json\n"
        "{\n"
        '  "prediction_prob": 0.0 to 1.0\n'
        "}\n"
        "```"
    )

    return instruction

def get_detailed_instruct_mortality_csv(config) -> str:
    cohort = str(config.get("mortality_cohort", "all")).strip().lower()
    cohort_context_map = {
        "micu": "Medical ICU (MICU)",
        "ccu": "Coronary Care Unit (CCU)",
        "all": "ICU",
    }
    # 1. Base Persona & Task
    instruction = (
        f"You are an expert Clinical Risk Estimation System analyzing a patient record from an emergent {cohort_context_map[cohort]} admission. "
        f"Your goal is to estimate the risk of in-{cohort_context_map[cohort]} mortality based on data collected during the first 48 hours of the {cohort_context_map[cohort]} stay.\n"
    )

    # 2. Structure for Free-Form Reasoning
    instruction += "\nPlease provide your analysis step-by-step using the following structure:\n"

    # Section A: Clinical Baseline
    instruction += (
        "\n1. CLINICAL ASSESSMENT: Analyze mortality risk based on the observed physiology (demographics, labs, vital signs etc.)."
    )

    # Section B: The Steering Block (Dynamic)
    if config["include_cot_prompt"]:
        instruction += (
            "\n2. MISSINGNESS MECHANISM: Analyze WHY specific features are missing. "
            "Consider whether their absence is potentially informative of the outcome."
        )
        # instruction += (
        #     "\n3. IMPUTATION: Estimate plausible values or clinical severity signals for "
        #     "missing features using the observed ones."
        # )

    # Section C: ICL Pattern Recognition
    if config["icl_num_examples"] > 0:
        if config["include_cot_prompt"]:
            instruction += (
                f"\n3. PATTERN RECOGNITION: Look at the few-shot examples provided from the hospital's {cohort_context_map[cohort]}. "
                "Identify any hospital-specific risk patterns and correlations using (a) observed values and (b) whether a feature is measured or not)."
            )
        else:
            instruction += (
                f"\n2. PATTERN RECOGNITION: Look at the few-shot examples provided from the hospital's {cohort_context_map[cohort]}. "
                "Identify any hospital-specific risk patterns and correlations using observed values."
            )

        if config["icl_sample_selection_mode"] == "balanced":
            instruction += (
                "\nNote: The few-shot examples are artificially balanced by class (positive and negative examples). "
            )

    if config['provide_prevalence'] == "baseline":
        if cohort == "micu":
            instruction += (
                "\nHowever, the true baseline mortality rate in this MICU is 12.4%. Please adjust your final probabilistic risk estimate to reflect this prevalence."
            )
        elif cohort == "ccu":
            instruction += (
                "\nHowever, the true baseline mortality rate in this CCU is 11.3%. Please adjust your final probabilistic risk estimate to reflect this prevalence."
            )

    # 3. Final Output Constraint (The JSON Anchor)
    instruction += (
        "\n\nAfter your analysis, you must output the final probability in a strictly valid JSON block "
        "at the very end of your response. Use this format:\n"
        "```json\n"
        "{\n"
        '  "prediction_prob": 0.0 to 1.0\n'
        "}\n"
        "```"
    )

    return instruction


def get_binary_instruct(config) -> str:
    query = config["task_query"]

    instruction = (
        "You are a helpful assistant that can answer questions about the patient's electronic health record."
    )
    if config["include_cot_prompt"]:
        instruction += (
            " In addition to the observed values, consider the missingness of recorded measurements "
            "as potentially informative. Explicitly reason about why certain measurements might be missing, "
            "and how their absence (not being measured) could affect your answer."
        )
        instruction += f"\nQuery: {query}"
    else:
        instruction += f"\nQuery: {query}"

    instruction += " Provide the final prediction as True or False at the end of your response in the following format: [Final Prediction: <prediction>]"
    return instruction

def extract_prediction(output: str) -> float:
    """
    Extract numeric prediction (0-1 float) from LLM output string,
    only if the prediction is explicitly given as a percentage.
    """
    try:
        # Require % sign
        matches = re.findall(r"Final Prediction:\s*([\d.]+)\s*%", output, re.IGNORECASE)
        if not matches:
            # Handle malformed bracketed variants like: [Final Prediction: 85]%
            matches = re.findall(
                r"\[\s*Final Prediction:\s*([\d.]+)\s*\]\s*%",
                output,
                re.IGNORECASE,
            )
        if not matches:
            return None

        pred_str = matches[-1]
        pred = float(pred_str) / 100.0
        return pred
    except Exception:
        return None


def extract_prediction_sim(output: str):
    """
    Extract numeric prediction (0-1 float) from a JSON response.
    Expected JSON contains a "prediction" field with a 0-100 value.
    """
    if output is None:
        return None

    def _coerce_prediction(value):
        try:
            pred = float(value)
        except Exception:
            return None
        if 0.0 <= pred <= 1.0:
            return pred
        if 1.0 < pred <= 100.0:
            return pred / 100.0
        return None

    def _try_json(text):
        try:
            obj = json.loads(text)
        except Exception:
            return None
        if isinstance(obj, dict):
            if "prediction_prob" in obj:
                return _coerce_prediction(obj.get("prediction_prob"))
            if "prediction" in obj:
                return _coerce_prediction(obj.get("prediction"))
        if isinstance(obj, (int, float)):
            return _coerce_prediction(obj)
        return None

    # Try fenced JSON blocks first
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", output, re.IGNORECASE)
    for block in fence_matches:
        pred = _try_json(block.strip())
        if pred is not None:
            return pred

    # Try any JSON object substring
    for match in re.findall(r"\{[\s\S]*?\}", output):
        pred = _try_json(match.strip())
        if pred is not None:
            return pred

    # Fallback: regex for prediction field
    regex_match = re.search(r"\"prediction_prob\"\s*:\s*([0-9]+(?:\.[0-9]+)?)", output)
    if not regex_match:
        regex_match = re.search(r"\"prediction\"\s*:\s*([0-9]+(?:\.[0-9]+)?)", output)
    if regex_match:
        return _coerce_prediction(regex_match.group(1))

    return None


def extract_binary_prediction(output: str):
    """
    Extract binary prediction (1 or 0) from LLM output string.
    Accepts True/False, Yes/No, or 1/0 after 'Final Prediction:'.
    """
    try:
        matches = re.findall(
            r"Final Prediction:\s*(true|false|yes|no|1|0)\b",
            output,
            re.IGNORECASE,
        )
        if not matches:
            return None

        value = matches[-1].lower()
        if value in {"true", "yes", "1"}:
            return 1
        if value in {"false", "no", "0"}:
            return 0
        return None
    except Exception:
        return None