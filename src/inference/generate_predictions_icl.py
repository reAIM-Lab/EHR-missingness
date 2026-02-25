import os
import json
import pickle
import random
from pathlib import Path

import polars as pl
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.inference.inference_utils import (
    get_detailed_instruct_mortality_csv,
    extract_prediction,
    serialize_mortality_csv_data,
)
from src.data.data_utils import (
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
        
    if "train" not in data_dicts or not data_dicts["train"]:
        raise ValueError("No training subjects available for ICL examples.")

    query = get_detailed_instruct_mortality_csv(config)
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
            for subject_id in unique_subjects
        ]

    # print(input_texts[0])

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
    max_model_len = int(config.get("max_model_len", 50000))
    max_tokens = int(config.get("max_tokens", 16000))
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

    # Only add "steered" to filename when steering is True; otherwise add nothing (no extra underscore).
    # Only applies the suffix if BOTH are true
    steering_part = "_steered" if config.get("steering") and config.get("include_cot_prompt") else ""

    data_seed = context_seed
    prediction_basename = (
        f"predictions_{config['downstream_task']}_{config['model_id'].split('/')[-1]}_"
        f"{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}{steering_part}_{icl_tag}"
        f"{cohort_tag}_seed{data_seed}{test_tag}"
    )
    if os.path.exists(predictions_dir / f"{prediction_basename}.pkl"):
        print(model_name, "ALREADY RUN")
        return True

    outputs_by_subject = [[] for _ in unique_subjects]

    if model_name.startswith("Qwen/"):
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
    steering_part = "_steered" if config.get("steering") else ""

    predictions_path = (
        predictions_dir
        / f"predictions_{config['downstream_task']}_{model_short_name}_"
        f"{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}{steering_part}_{icl_tag}"
        f"{cohort_tag}_seed{data_seed}{test_tag}.pkl"
    )

    with open(predictions_path, "wb") as f:
        pickle.dump(subject_dicts, f)
