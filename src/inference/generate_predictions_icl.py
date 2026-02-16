import os
import json
import pickle
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import polars as pl
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from anthropic import Anthropic

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
)

from src.inference.inference_utils import (
    get_detailed_instruct,
    get_detailed_instruct_sim,
    get_detailed_instruct_mortality_csv,
    extract_prediction,
    extract_prediction_sim,
    serialize_data,
    serialize_sim_data,
    serialize_mortality_csv_data,
)
from src.data.data_utils import (
    sample_df,
    generate_sepsis_cohort,
    load_mortality_csv_with_splits,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _sample_stratified_ids(subject_dicts_train, num_examples, rng):
    label_map = {0: [], 1: []}
    for subject_id, payload in subject_dicts_train.items():
        label = payload.get("label")
        try:
            label_int = int(label)
        except Exception:
            label_int = 1 if str(label).strip().lower() in ("true", "1", "yes") else 0
        label_map[1 if label_int == 1 else 0].append(subject_id)

    total_ids = label_map[0] + label_map[1]
    if not total_ids:
        return []

    k = min(num_examples, len(total_ids))
    if not label_map[0] or not label_map[1]:
        return rng.sample(total_ids, k)

    prevalence = len(label_map[1]) / len(total_ids)
    k_pos = int(round(k * prevalence))
    k_neg = k - k_pos

    # ensure at least 1 from each class when possible
    if k >= 2:
        k_pos = max(1, k_pos)
        k_neg = max(1, k - k_pos)
        if k_pos + k_neg != k:
            k_neg = k - k_pos

    k_pos = min(k_pos, len(label_map[1]))
    k_neg = min(k_neg, len(label_map[0]))

    sampled_pos = rng.sample(label_map[1], k_pos)
    sampled_neg = rng.sample(label_map[0], k_neg)
    sampled_ids = sampled_pos + sampled_neg

    # top up if rounding or class size limited
    if len(sampled_ids) < k:
        remaining = [sid for sid in total_ids if sid not in sampled_ids]
        sampled_ids += rng.sample(remaining, k - len(sampled_ids))

    rng.shuffle(sampled_ids)
    return sampled_ids

def _sample_balanced_ids(subject_dicts_train, num_examples, rng):
    label_map = {0: [], 1: []}
    for subject_id, payload in subject_dicts_train.items():
        label = payload.get("label")
        try:
            label_int = int(label)
        except Exception:
            label_int = 1 if str(label).strip().lower() in ("true", "1", "yes") else 0
        label_map[1 if label_int == 1 else 0].append(subject_id)

    total_ids = label_map[0] + label_map[1]
    if not total_ids:
        return []

    k = min(num_examples, len(total_ids))
    if not label_map[0] or not label_map[1]:
        return rng.sample(total_ids, k)

    k_pos = min(len(label_map[1]), k // 2)
    k_neg = min(len(label_map[0]), k // 2)

    # For odd k, assign the extra slot to whichever class has more available.
    if k % 2 == 1:
        if len(label_map[1]) >= len(label_map[0]) and k_pos < len(label_map[1]):
            k_pos += 1
        elif k_neg < len(label_map[0]):
            k_neg += 1
        elif k_pos < len(label_map[1]):
            k_pos += 1

    sampled_pos = rng.sample(label_map[1], k_pos)
    sampled_neg = rng.sample(label_map[0], k_neg)
    sampled_ids = sampled_pos + sampled_neg

    # Top up if one class is too small.
    if len(sampled_ids) < k:
        remaining = [sid for sid in total_ids if sid not in sampled_ids]
        sampled_ids += rng.sample(remaining, k - len(sampled_ids))

    rng.shuffle(sampled_ids)
    return sampled_ids

def _format_icl_examples(
    subject_dicts_train, num_examples, rng, pool_size=None, selection_mode=None
):
    subject_ids = list(subject_dicts_train.keys())
    if not subject_ids:
        return ""

    if pool_size is None:
        pool_size = num_examples

    if selection_mode is None:
        selection_mode = "random"
    selection_mode = str(selection_mode).strip().lower()

    if selection_mode == "balanced":
        sampled_ids = _sample_balanced_ids(subject_dicts_train, pool_size, rng)
    elif selection_mode == "stratified":
        sampled_ids = _sample_stratified_ids(subject_dicts_train, pool_size, rng)
    elif selection_mode == "random":
        k = min(pool_size, len(subject_ids))
        sampled_ids = rng.sample(subject_ids, k)
    else:
        raise ValueError(
            f"Unsupported icl_sample_selection_mode: {selection_mode}. "
            "Expected one of: random, stratified, balanced."
        )

    if num_examples < len(sampled_ids):
        sampled_ids = sampled_ids[:num_examples]

    example_blocks = []
    for i, subject_id in enumerate(sampled_ids, start=1):
        serialization = subject_dicts_train[subject_id]["serialization"]
        label = subject_dicts_train[subject_id]["label"]
        try:
            label_bool = bool(int(label))
        except Exception:
            label_bool = str(label).strip().lower() in ("true", "1", "yes")

        example_blocks.append(
            "### Example {idx}\n{serialization}\nAnswer: {label}\n".format(
                idx=i, serialization=serialization, label="True" if label_bool else "False"
            )
        )

    return "\n".join(example_blocks).strip()


def _build_icl_prompt(instruction, examples_block, target_serialization):
    prompt = instruction.strip()
    if examples_block:
        prompt += "\n\nHere are some labeled examples from this hospital:\n"
        prompt += examples_block
        # prompt += "\n\nNow output response for the next patient.\n"
        prompt += "\n\n Now output response for the next patient using BOTH:"
        prompt += "\n(A) general clinical knowledge about risk factors, and (B) hospital-specific patterns inferred from the labeled examples.\n"
    prompt += "\n### Patient\n"
    prompt += target_serialization
    prompt += "\n### Response:\n"
    return prompt

def _build_icl_prompt_sim(instruction, examples_block, target_serialization):
    prompt = instruction.strip()
    if examples_block:
        prompt += "\n\nHere are some labeled examples from this hospital:\n"
        prompt += examples_block
        prompt += "\n\nNow output response for the next patient.\n"
    prompt += "\n### Patient\n"
    prompt += target_serialization
    prompt += "\n### Response:\n"
    return prompt


def _get_mortality_cohort(config):
    return str(config.get("mortality_cohort", "all")).strip().lower()


def _split_prompt_for_anthropic_cache(prompt):
    """
    Split prompt into a stable prefix and per-patient suffix for Anthropic prompt caching.
    The prefix includes instruction + ICL examples (if present), while the suffix starts at patient data.
    """
    marker = "\n### Patient\n"
    marker_idx = prompt.find(marker)
    if marker_idx == -1:
        return prompt, "", False
    prefix = prompt[:marker_idx]
    suffix = prompt[marker_idx:]
    return prefix, suffix, bool(prefix.strip())


def _sdk_obj_to_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {}


def _extract_anthropic_text_from_message(message_obj):
    if hasattr(message_obj, "content"):
        text_blocks = []
        for block in getattr(message_obj, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_blocks.append(getattr(block, "text", ""))
        return "\n".join([t for t in text_blocks if t]).strip()

    message_dict = _sdk_obj_to_dict(message_obj)
    text_blocks = [
        block.get("text", "")
        for block in message_dict.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join([t for t in text_blocks if t]).strip()


def _extract_anthropic_usage(usage_obj):
    usage_dict = _sdk_obj_to_dict(usage_obj)
    return {
        "input_tokens": int(usage_dict.get("input_tokens", 0) or 0),
        "output_tokens": int(usage_dict.get("output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(usage_dict.get("cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(usage_dict.get("cache_read_input_tokens", 0) or 0),
    }

def _write_vertex_batch_jsonl(
    batch_jsonl_path,
    batch_index_path,
    input_texts,
    unique_subjects,
    num_generations,
    temperature,
    top_p,
    max_tokens,
):
    generation_config = {
        "temperature": temperature,
        "topP": top_p,
        "maxOutputTokens": max_tokens,
    }
    rows_written = 0
    batch_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_jsonl_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(input_texts):
            subject_id = unique_subjects[i]
            for sample_idx in range(num_generations):
                request_obj = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": prompt}],
                            }
                        ],
                        "generationConfig": generation_config,
                    }
                }
                f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
                rows_written += 1

    index_payload = {
        "rows_written": rows_written,
        "num_subjects": len(unique_subjects),
        "num_generations": num_generations,
        "index_format": "line_idx -> {subject_id, sample_idx}",
        "line_index": [],
    }
    line_idx = 0
    for i, subject_id in enumerate(unique_subjects):
        _ = i
        for sample_idx in range(num_generations):
            index_payload["line_index"].append(
                {
                    "line_idx": line_idx,
                    "subject_id": subject_id,
                    "sample_idx": sample_idx,
                }
            )
            line_idx += 1

    with open(batch_index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)


def _extract_sim_subject_metadata(subject_data):
    """
    Capture all available SIM per-subject fields from a single grouped row.
    Polars grouped subject_data is expected to contain one row per subject.
    """
    metadata = {}
    for col in subject_data.columns:
        try:
            value = subject_data[col][0]
        except Exception:
            continue
        # Convert numpy scalars for safe pickling/JSON interoperability downstream.
        if isinstance(value, np.generic):
            value = value.item()
        metadata[col] = value
    return metadata


def generate_predictions_icl(config):
    experiment = config.get("experiment")
    cohort_tag = ""
    if experiment == "mortality_csv":
        cohort_tag = f"_cohort-{_get_mortality_cohort(config)}"

    sim_split_seed = int(config.get("split_seed", config.get("data_seed", 0)))
    sim_seed_tag = f"_splitseed-{sim_split_seed}" if experiment == "sim" else ""
    subject_dict_path = (
        Path(config["target_dir"])
        / (
            f"{config['downstream_task']}_subject_dict_"
            f"{config['explicit_missingness']}_{config['labs_only']}{cohort_tag}{sim_seed_tag}.pkl"
        )
    )

    if subject_dict_path.exists():
        with open(subject_dict_path, "rb") as f:
            data_dicts = pickle.load(f)
        print(f"Loaded subject_dicts from {subject_dict_path}")
    else:
        if experiment == "sim":
            n_samples = int(config.get("sim_samples", config.get("max_samples", 1000)))
            np.random.seed(sim_split_seed)

            sim_df = generate_sepsis_cohort(n=n_samples).copy()
            sim_df["subject_id"] = list(range(n_samples))
            sim_df["true_prob"] = 1 / (1 + np.exp(-sim_df["True_logits"]))

            rng = random.Random(sim_split_seed)
            subject_ids = list(sim_df["subject_id"])
            rng.shuffle(subject_ids)
            train_frac = float(config.get("train_frac", 0.5))
            split_idx = int(round(train_frac * len(subject_ids)))
            train_ids = set(subject_ids[:split_idx])
            sim_df["split"] = [
                "train" if subject_id in train_ids else "eval"
                for subject_id in sim_df["subject_id"]
            ]

            sim_data = pl.from_pandas(sim_df)
            train_data = sim_data.filter(pl.col("split") == "train")
            test_data = sim_data.filter(pl.col("split") == "eval")

            data_dicts = {}

            subject_groups = test_data.partition_by("subject_id", as_dict=True)
            subject_dicts_test = {}
            for subject_id, subject_data in subject_groups.items():
                serialization = serialize_sim_data(subject_data, config)
                label = subject_data["boolean_value"][0]
                metadata = _extract_sim_subject_metadata(subject_data)
                subject_dicts_test[subject_id[0]] = {
                    "serialization": serialization,
                    "label": label,
                    "metadata": metadata,
                }

            data_dicts["eval"] = subject_dicts_test

            subject_groups = train_data.partition_by("subject_id", as_dict=True)
            subject_dicts_train = {}
            for subject_id, subject_data in subject_groups.items():
                serialization = serialize_sim_data(subject_data, config)
                label = subject_data["boolean_value"][0]
                metadata = _extract_sim_subject_metadata(subject_data)
                subject_dicts_train[subject_id[0]] = {
                    "serialization": serialization,
                    "label": label,
                    "metadata": metadata,
                }

            data_dicts["train"] = subject_dicts_train

            with open(subject_dict_path, "wb") as f:
                pickle.dump(data_dicts, f)

            print(f"Saved subject_dicts to {subject_dict_path}")
        elif experiment == "mortality_csv":
            test_data, train_data = load_mortality_csv_with_splits(config)
            data_dicts = {}

            subject_groups = test_data.partition_by("subject_id", as_dict=True)
            subject_dicts_test = {}
            for subject_id, subject_data in subject_groups.items():
                serialization = serialize_mortality_csv_data(subject_data, config)
                label = subject_data["boolean_value"][0]
                subject_dicts_test[subject_id[0]] = {
                    "serialization": serialization,
                    "label": label,
                }

            data_dicts["eval"] = subject_dicts_test

            subject_groups = train_data.partition_by("subject_id", as_dict=True)
            subject_dicts_train = {}
            for subject_id, subject_data in subject_groups.items():
                serialization = serialize_mortality_csv_data(subject_data, config)
                label = subject_data["boolean_value"][0]
                subject_dicts_train[subject_id[0]] = {
                    "serialization": serialization,
                    "label": label,
                }

            data_dicts["train"] = subject_dicts_train

            with open(subject_dict_path, "wb") as f:
                pickle.dump(data_dicts, f)

            print(f"Saved subject_dicts to {subject_dict_path}")
        else:
            new_meds_dir = Path(config["target_dir"]) / f"{config['downstream_task']}_MEDS"
            train_files = sorted((new_meds_dir / "data" / "train").glob("*.parquet"))
            tune_files = sorted((new_meds_dir / "data" / "tuning").glob("*.parquet"))
            test_files = sorted((new_meds_dir / "data" / "held_out").glob("*.parquet"))

            # Load and concatenate all data
            dfs = []
            for files in [train_files, tune_files, test_files]:
                for file in files:
                    df = pl.read_parquet(file)
                    dfs.append(df)

            data = pl.concat(dfs)
            test_data, train_data = sample_df(data, config)

            data_dicts = {}

            subject_groups = test_data.partition_by("subject_id", as_dict=True)
            subject_dicts_test = {}
            for subject_id, subject_data in subject_groups.items():
                serialization = serialize_data(subject_data, config)
                label = subject_data["boolean_value"][0]
                subject_dicts_test[subject_id[0]] = {
                    "serialization": serialization,
                    "label": label,
                }

            data_dicts["eval"] = subject_dicts_test

            subject_groups = train_data.partition_by("subject_id", as_dict=True)
            subject_dicts_train = {}
            for subject_id, subject_data in subject_groups.items():
                serialization = serialize_data(subject_data, config)
                label = subject_data["boolean_value"][0]
                subject_dicts_train[subject_id[0]] = {
                    "serialization": serialization,
                    "label": label,
                }

            data_dicts["train"] = subject_dicts_train

            # Save subject_dicts to target directory
            with open(subject_dict_path, "wb") as f:
                pickle.dump(data_dicts, f)

            print(f"Saved subject_dicts to {subject_dict_path}")

    if "train" not in data_dicts or not data_dicts["train"]:
        raise ValueError("No training subjects available for ICL examples.")

    if experiment == "sim":
        query = get_detailed_instruct_sim(config)
    elif experiment == "mortality_csv":
        query = get_detailed_instruct_mortality_csv(config)
    else:
        query = get_detailed_instruct(config)
    
    subject_dicts = data_dicts.get("eval", data_dicts)
    unique_subjects = list(subject_dicts.keys())

    # data_seed remains the per-run/context sampling seed (set in main.py loops),
    # while split_seed is reserved for reproducible SIM dataset generation/splitting.
    context_seed = int(config.get("data_seed", 0))
    rng = random.Random(context_seed)
    icl_num_examples = int(config.get("icl_num_examples", 20))
    icl_pool_size = int(config.get("icl_pool_size", icl_num_examples))
    icl_selection_mode = str(config.get("icl_sample_selection_mode", "random")).strip().lower()
    provide_prevalence = str(config.get("provide_prevalence", "baseline")).strip().lower()

    examples_block = _format_icl_examples(
        data_dicts["train"],
        icl_num_examples,
        rng,
        pool_size=icl_pool_size,
        selection_mode=icl_selection_mode,
    )

    if icl_num_examples == 0:
        input_texts = [
            query + subject_dicts[subject_id]["serialization"]
            for subject_id in unique_subjects
        ]
    else:
        input_texts = [
            _build_icl_prompt_sim(query, examples_block, subject_dicts[subject_id]["serialization"])
            if config.get("experiment") in {"sim", "mortality_csv"}
            else _build_icl_prompt(query, examples_block, subject_dicts[subject_id]["serialization"])
            for subject_id in unique_subjects
        ]

    test_num_samples = int(config.get("test_num_samples", 0) or 0)
    test_mode = test_num_samples > 0
    test_tag = ""
    if test_mode:
        test_num_samples = min(test_num_samples, len(input_texts))
        unique_subjects = unique_subjects[:test_num_samples]
        input_texts = input_texts[:test_num_samples]
        test_tag = f"_test{test_num_samples}"
        print(f"[TEST MODE] Running only {test_num_samples} samples")

    cache_dir = config.get("cache_dir")
    model_name = config["model_id"]
    default_max_model_len = 50000 if experiment == "mimic" else 10000 if experiment == "sim" else 50000
    default_max_tokens = 16000 if experiment == "mimic" else 5000 if experiment == "sim" else 16000
    max_model_len = int(config.get("max_model_len", default_max_model_len))
    max_tokens = int(config.get("max_tokens", default_max_tokens))
    num_generations = int(config.get("num_generations", config.get("num_samples", 5)))
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1")
    default_temperature = 0.7 if num_generations > 1 else 0.0
    temperature = float(config.get("temperature", default_temperature))
    top_p = float(config.get("top_p", 0.95 if temperature > 0 else 1.0))
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")
    if num_generations > 1 and temperature == 0.0:
        print(
            "[WARN] num_generations > 1 with temperature=0.0; samples may be identical. "
            "Set temperature > 0 for stochastic generation."
        )
    print(
        f"Generation config: num_generations={num_generations}, "
        f"temperature={temperature}, top_p={top_p}"
    )
    predictions_dir = Path(config["predictions_dir"])
    if icl_num_examples > 0:
        icl_tag = f"icl{icl_num_examples}_{icl_selection_mode}"
    else:
        icl_tag = (
            f"icl{icl_num_examples}_{provide_prevalence}"
            if provide_prevalence == "baseline"
            else f"icl{icl_num_examples}"
        )
    data_seed = context_seed
    prediction_basename = (
        f"predictions_{config['downstream_task']}_{config['model_id'].split('/')[-1]}_"
        f"{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}_{icl_tag}"
        f"{cohort_tag}_seed{data_seed}{test_tag}"
    )
    if os.path.exists(predictions_dir / f"{prediction_basename}.pkl"):
        print(model_name, "ALREADY RUN")
        return True
    
    export_vertex_batch_jsonl = bool(config.get("vertex_export_batch_jsonl", False))
    if export_vertex_batch_jsonl and (
        model_name.startswith("google/gemini") or model_name.startswith("gemini-")
    ):
        batch_input_dir = Path(
            config.get(
                "vertex_batch_input_dir",
                str(Path(config["predictions_dir"]) / "vertex_batch_inputs"),
            )
        )
        batch_jsonl_path = batch_input_dir / f"{prediction_basename}.jsonl"
        batch_index_path = batch_input_dir / f"{prediction_basename}.index.json"
        _write_vertex_batch_jsonl(
            batch_jsonl_path=batch_jsonl_path,
            batch_index_path=batch_index_path,
            input_texts=input_texts,
            unique_subjects=unique_subjects,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        print("Vertex batch JSONL written:", batch_jsonl_path)
        print("Vertex batch row index written:", batch_index_path)
        print(
            f"Rows={len(unique_subjects) * num_generations} "
            f"(subjects={len(unique_subjects)} x generations={num_generations})"
        )
        return True

    outputs_by_subject = [[] for _ in unique_subjects]

    if model_name.startswith("anthropic/") or model_name.startswith("claude-"):
        api_key = (
            config.get("anthropic_key")
            or config.get("key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Missing Anthropic API key. Set config['anthropic_key'] (or config['key']) "
                "or export ANTHROPIC_API_KEY."
            )

        model_alias = model_name.split("/")[-1]
        anthropic_model = config.get(
            "anthropic_model",
            {
                "sonnet-4.5": "claude-sonnet-4-5-20250929",
            }.get(model_alias, model_alias),
        )
        client = Anthropic(api_key=api_key)
        batch_id = config.get("id", None)
        use_batch = bool(config.get("anthropic_use_batch", True))
        test_cache = bool(config.get("anthropic_test_cache", False))
        anthropic_usage_rows = []

        if test_cache and use_batch:
            # Cache hit diagnostics are easier to validate in sequential requests.
            print("[TEST MODE] Forcing anthropic_use_batch=False to validate cache behavior")
            use_batch = False

        if use_batch:
            if batch_id is None:
                requests_payload = []
                for i, prompt in enumerate(input_texts):
                    prefix, suffix, can_cache_prefix = _split_prompt_for_anthropic_cache(prompt)
                    user_content = []
                    if can_cache_prefix:
                        user_content.append(
                            {
                                "type": "text",
                                "text": prefix,
                                "cache_control": {"type": "ephemeral"},
                            }
                        )
                        if suffix:
                            user_content.append({"type": "text", "text": suffix})
                    else:
                        user_content.append({"type": "text", "text": prompt})

                    for sample_idx in range(num_generations):
                        requests_payload.append(
                            {
                                "custom_id": f"{i}__{sample_idx}",
                                "params": {
                                    "model": anthropic_model,
                                    "max_tokens": max_tokens,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "messages": [{"role": "user", "content": user_content}],
                                },
                            }
                        )

                response_data = client.messages.batches.create(requests=requests_payload)
                print("Anthropic batch submitted")
                print("-" * 42)
                print("Batch ID:", response_data.id)
                return None

            batch_meta = client.messages.batches.retrieve(batch_id)
            print(batch_meta)
            processing_status = getattr(batch_meta, "processing_status", None)
            print(f"Status: {processing_status}")
            if processing_status != "ended":
                print("Batch unfinished!")
                return None
            print("Batch completed!")

            resp_map = {}
            for row in client.messages.batches.results(batch_id):
                custom_id_raw = str(getattr(row, "custom_id", "-1"))
                if "__" in custom_id_raw:
                    subj_str, sample_str = custom_id_raw.split("__", 1)
                    custom_id = int(subj_str)
                    sample_idx = int(sample_str)
                else:
                    custom_id = int(custom_id_raw)
                    sample_idx = 0
                result = getattr(row, "result", None)
                result_type = getattr(result, "type", None)
                if result_type == "succeeded":
                    message_obj = result.message
                    if custom_id not in resp_map:
                        resp_map[custom_id] = {}
                    resp_map[custom_id][sample_idx] = _extract_anthropic_text_from_message(message_obj)
                    if test_mode:
                        usage = _extract_anthropic_usage(getattr(message_obj, "usage", {}))
                        usage["custom_id"] = f"{custom_id}__{sample_idx}"
                        anthropic_usage_rows.append(usage)
                elif result_type == "errored":
                    raise RuntimeError(
                        f"Anthropic batch item {custom_id} errored: "
                        f"{json.dumps(_sdk_obj_to_dict(getattr(result, 'error', {})))}"
                    )
                else:
                    raise RuntimeError(
                        f"Anthropic batch item {custom_id} returned non-success status: {result_type}"
                    )

            missing_ids = [i for i in range(len(input_texts)) if i not in resp_map]
            if missing_ids:
                raise RuntimeError(
                    f"Anthropic batch completed but missing results for IDs: {missing_ids[:10]}"
                )
            for cid in sorted(resp_map):
                sample_map = resp_map[cid]
                sample_texts = [sample_map[sid] for sid in sorted(sample_map)]
                outputs_by_subject[cid] = sample_texts[:num_generations]
        else:
            outputs_by_subject = [[] for _ in unique_subjects]
            for i, prompt in enumerate(input_texts):
                prefix, suffix, can_cache_prefix = _split_prompt_for_anthropic_cache(prompt)
                user_content = []
                if can_cache_prefix:
                    user_content.append(
                        {
                            "type": "text",
                            "text": prefix,
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
                    if suffix:
                        user_content.append({"type": "text", "text": suffix})
                else:
                    user_content.append({"type": "text", "text": prompt})

                for sample_idx in range(num_generations):
                    payload = {
                        "model": anthropic_model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "messages": [{"role": "user", "content": user_content}],
                    }
                    response_data = client.messages.create(**payload)
                    if test_mode:
                        usage = _extract_anthropic_usage(getattr(response_data, "usage", {}))
                        usage["custom_id"] = f"{i}__{sample_idx}"
                        anthropic_usage_rows.append(usage)
                    outputs_by_subject[i].append(_extract_anthropic_text_from_message(response_data))

        if test_mode and anthropic_usage_rows:
            total_input_tokens = sum(row["input_tokens"] for row in anthropic_usage_rows)
            total_output_tokens = sum(row["output_tokens"] for row in anthropic_usage_rows)
            total_cache_write = sum(row["cache_creation_input_tokens"] for row in anthropic_usage_rows)
            total_cache_read = sum(row["cache_read_input_tokens"] for row in anthropic_usage_rows)
            cache_read_hits = sum(1 for row in anthropic_usage_rows if row["cache_read_input_tokens"] > 0)
            print(
                "[TEST MODE] Anthropic usage totals:"
                f" input={total_input_tokens}, output={total_output_tokens}, "
                f"cache_write={total_cache_write}, cache_read={total_cache_read}"
            )
            print(
                "[TEST MODE] Cache-read hits:"
                f" {cache_read_hits}/{len(anthropic_usage_rows)} requests"
            )
            for row in anthropic_usage_rows:
                print(
                    "[TEST MODE] sample"
                    f" {row['custom_id']}: input={row['input_tokens']}, output={row['output_tokens']}, "
                    f"cache_write={row['cache_creation_input_tokens']}, "
                    f"cache_read={row['cache_read_input_tokens']}"
                )
    elif "openai/gpt-oss" in model_name:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        convos = [
            Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                    Message.from_role_and_content(Role.USER, prompt),
                ]
            )
            for prompt in input_texts
        ]

        prefill_ids = [
            encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            for convo in convos
        ]
        inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prefill_ids]

        # Harmony stop tokens (pass to sampler so they won't be included in output)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            top_p=top_p,
            n=num_generations,
        )

        # Initialize the vLLM engine
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
            enable_prefix_caching=True,
        )

        outputs = llm.generate(
            prompts=inputs,
            sampling_params=sampling_params,
        )
        for i, output in enumerate(outputs):
            outputs_by_subject[i] = [candidate.text for candidate in output.outputs][:num_generations]

    elif model_name.startswith("Qwen/"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cache_dir)
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
            enable_prefix_caching=True,
            # hf_overrides={
            #     "rope_parameters": {
            #         "rope_type": "yarn",
            #         "factor": 4.0,
            #         "original_max_position_embeddings": 32768,
            #     }
            # }
        )

        stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids or None,
            top_p=top_p,
            n=num_generations,
        )

        messages = [{"role": "user", "content": prompt} for prompt in input_texts]
        texts = [
            tokenizer.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message in messages
        ]

        outputs = llm.generate(texts, sampling_params)
        for i, output in enumerate(outputs):
            outputs_by_subject[i] = [candidate.text for candidate in output.outputs][:num_generations]

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cache_dir)
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
            enable_prefix_caching=True,
        )

        messages = [{"role": "user", "content": prompt} for prompt in input_texts]

        texts = [
            tokenizer.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]

        stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids or None,
            top_p=top_p,
            n=num_generations,
        )
        outputs = llm.generate(texts, sampling_params)
        for i, output in enumerate(outputs):
            outputs_by_subject[i] = [candidate.text for candidate in output.outputs][:num_generations]

    # Ensure each subject has the expected number of generations.
    missing_generation_ids = [
        unique_subjects[i]
        for i, sample_texts in enumerate(outputs_by_subject)
        if len(sample_texts) < num_generations
    ]
    if missing_generation_ids:
        raise RuntimeError(
            "Missing generated samples for some subjects. "
            f"Expected {num_generations} each, missing for up to first 10 IDs: "
            f"{missing_generation_ids[:10]}"
        )

    # Collect results
    for i, sample_texts in enumerate(outputs_by_subject):
        sample_preds = []
        for output_text in sample_texts:
            if config.get("experiment") in {"sim", "mortality_csv"}:
                pred = extract_prediction_sim(output_text)
            else:
                pred = extract_prediction(output_text)
            sample_preds.append(pred)
        subject_id = unique_subjects[i]
        subject_dicts[subject_id]["response"] = sample_texts[0]
        subject_dicts[subject_id]["prediction"] = sample_preds[0]
        subject_dicts[subject_id]["response_samples"] = sample_texts
        subject_dicts[subject_id]["prediction_samples"] = sample_preds
        if test_mode:
            label = subject_dicts[subject_id].get("label")
            print(
                f"[TEST MODE] subject={subject_id} label={label} "
                f"predictions={sample_preds} response_chars={[len(t) for t in sample_texts]}"
            )
            for sample_idx, response_text in enumerate(sample_texts):
                print(
                    f"[TEST MODE] subject={subject_id} sample={sample_idx} "
                    f"response:\n{response_text}"
                )

    # Save predictions as pickle file
    predictions_dir = Path(config["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_short_name = config["model_id"].split("/")[-1]

    predictions_path = (
        predictions_dir
        / f"predictions_{config['downstream_task']}_{model_short_name}_"
        f"{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}_{icl_tag}"
        f"{cohort_tag}_seed{data_seed}{test_tag}.pkl"
    )

    with open(predictions_path, "wb") as f:
        pickle.dump(subject_dicts, f)
