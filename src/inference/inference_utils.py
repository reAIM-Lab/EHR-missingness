import re
import json
import math
import polars as pl

from src.data.data_utils import extract_demo
from src.data.feature_configs import codes_to_keep, codes_to_keep_mimic

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
        if config['steering']:
            instruction += (
                "\n2. MISSINGNESS MECHANISM: Recognize that missing values reflect a clinician's decision that the patient is stable. "
                "Use the absence of measurements as a protective signal."
            )
        else:
            instruction += (
                "\n2. MISSINGNESS MECHANISM: Analyze WHY specific features are missing. "
                "Consider whether their absence is potentially informative of the outcome."
            )

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



def extract_prediction(output: str):
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
