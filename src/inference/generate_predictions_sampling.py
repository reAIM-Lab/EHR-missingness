import os
import pickle
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
)

from src.inference.inference_utils import (
    get_binary_instruct,
    extract_binary_prediction,
    get_detailed_instruct,
    get_detailed_instruct_sim,
    extract_prediction,
    extract_prediction_sim,
    serialize_data,
)
from src.data.data_utils import sample_df


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def generate_predictions_sampling(config):
    # Load in preprocessed MEDS data

    subject_dict_path = (
        Path(config["target_dir"])
        / f"{config['downstream_task']}_subject_dict_{config['explicit_missingness']}_{config['labs_only']}.pkl"
    )

    if subject_dict_path.exists():
        with open(subject_dict_path, "rb") as f:
            subject_dicts = pickle.load(f)
        print(f"Loaded subject_dicts from {subject_dict_path}")
        # If this was created by ICL, it contains "train"/"eval" splits.
        if isinstance(subject_dicts, dict) and "eval" in subject_dicts:
            subject_dicts = subject_dicts["eval"]
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
        data = sample_df(data, config)

        subject_groups = data.partition_by("subject_id", as_dict=True)
        subject_dicts = {}
        for subject_id, subject_data in subject_groups.items():
            serialization = serialize_data(subject_data, config)
            label = subject_data["boolean_value"][0]
            subject_dicts[subject_id[0]] = {
                "serialization": serialization,
                "label": label,
            }

        # Save subject_dicts to target directory
        with open(subject_dict_path, "wb") as f:
            pickle.dump(subject_dicts, f)

        print(f"Saved subject_dicts to {subject_dict_path}")

    query = get_binary_instruct(config)
    unique_subjects = list(subject_dicts.keys())
    input_texts = [query + subject_dicts[subject_id]["serialization"] for subject_id in unique_subjects]

    cache_dir = config.get("cache_dir")
    model_name = config["model_id"]
    experiment = config.get("experiment")
    default_max_model_len = 50000 if experiment == "mimic" else 8000 if experiment == "sim" else 50000
    default_max_tokens = 16000 if experiment == "mimic" else 2000 if experiment == "sim" else 16000
    max_model_len = int(config.get("max_model_len", default_max_model_len))
    max_tokens = int(config.get("max_tokens", default_max_tokens))
    num_samples = int(config.get("num_samples", 10))
    temperature = float(config.get("sampling_temperature", 0.7))

    if "openai/gpt-oss" in model_name:
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
            n=num_samples,
            stop_token_ids=stop_token_ids,
        )

        # Initialize the vLLM engine
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
        )

        outputs = llm.generate(
            prompts=inputs,
            sampling_params=sampling_params,
        )

    elif model_name.startswith("Qwen/"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            cache_dir=cache_dir,
        )
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
        )

        stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_samples,
            stop_token_ids=stop_token_ids or None,
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

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            cache_dir=cache_dir,
        )
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
        )

        stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_samples,
            stop_token_ids=stop_token_ids or None,
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

        outputs = llm.generate(texts, sampling_params)

    # Collect results
    for i, output in enumerate(outputs):
        output_texts = [sample.text for sample in output.outputs]
        preds = [extract_binary_prediction(text) for text in output_texts]

        subject_id = unique_subjects[i]
        subject_dicts[subject_id]["responses"] = output_texts
        subject_dicts[subject_id]["predictions"] = preds
        subject_dicts[subject_id]["num_samples"] = num_samples
        subject_dicts[subject_id]["sampling_temperature"] = temperature

    # Save predictions as pickle file
    predictions_dir = Path(config["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["model_id"].split("/")[-1]
    temp_tag = str(temperature).replace(".", "p")
    predictions_path = (
        predictions_dir
        / f"predictions_sampling_{config['downstream_task']}_{model_name}_{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}_n{num_samples}_t{temp_tag}.pkl"
    )

    with open(predictions_path, "wb") as f:
        pickle.dump(subject_dicts, f)


def generate_predictions_sampling_probability(config):
    # Load in preprocessed MEDS data
    subject_dict_path = (
        Path(config["target_dir"])
        / f"{config['downstream_task']}_subject_dict_{config['explicit_missingness']}_{config['labs_only']}.pkl"
    )

    if subject_dict_path.exists():
        with open(subject_dict_path, "rb") as f:
            subject_dicts = pickle.load(f)
        print(f"Loaded subject_dicts from {subject_dict_path}")
        # If this was created by ICL, it contains "train"/"eval" splits.
        if isinstance(subject_dicts, dict) and "eval" in subject_dicts:
            subject_dicts = subject_dicts["eval"]
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
        data = sample_df(data, config)

        subject_groups = data.partition_by("subject_id", as_dict=True)
        subject_dicts = {}
        for subject_id, subject_data in subject_groups.items():
            serialization = serialize_data(subject_data, config)
            label = subject_data["boolean_value"][0]
            subject_dicts[subject_id[0]] = {
                "serialization": serialization,
                "label": label,
            }

        # Save subject_dicts to target directory
        with open(subject_dict_path, "wb") as f:
            pickle.dump(subject_dicts, f)

        print(f"Saved subject_dicts to {subject_dict_path}")

    if config.get("experiment") == "sim":
        query = get_detailed_instruct_sim(config)
    else:
        query = get_detailed_instruct(config)
    unique_subjects = list(subject_dicts.keys())
    input_texts = [query + subject_dicts[subject_id]["serialization"] for subject_id in unique_subjects]

    cache_dir = config.get("cache_dir")
    model_name = config["model_id"]
    experiment = config.get("experiment")
    default_max_model_len = 50000 if experiment == "mimic" else 8000 if experiment == "sim" else 50000
    default_max_tokens = 16000 if experiment == "mimic" else 2000 if experiment == "sim" else 16000
    max_model_len = int(config.get("max_model_len", default_max_model_len))
    max_tokens = int(config.get("max_tokens", default_max_tokens))
    num_samples = int(config.get("num_samples", 10))
    temperature = float(config.get("sampling_temperature", 0.7))

    if "openai/gpt-oss" in model_name:
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
            n=num_samples,
            stop_token_ids=stop_token_ids,
        )

        # Initialize the vLLM engine
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
        )

        outputs = llm.generate(
            prompts=inputs,
            sampling_params=sampling_params,
        )

    elif model_name.startswith("Qwen/"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            cache_dir=cache_dir,
        )
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
        )

        stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_samples,
            stop_token_ids=stop_token_ids or None,
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

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            cache_dir=cache_dir,
        )
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            download_dir=cache_dir,
        )

        stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_samples,
            stop_token_ids=stop_token_ids or None,
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

        outputs = llm.generate(texts, sampling_params)

    # Collect results
    for i, output in enumerate(outputs):
        output_texts = [sample.text for sample in output.outputs]
        if config.get("experiment") == "sim":
            preds = [extract_prediction_sim(text) for text in output_texts]
        else:
            preds = [extract_prediction(text) for text in output_texts]

        subject_id = unique_subjects[i]
        subject_dicts[subject_id]["responses"] = output_texts
        subject_dicts[subject_id]["predictions"] = preds
        subject_dicts[subject_id]["num_samples"] = num_samples
        subject_dicts[subject_id]["sampling_temperature"] = temperature

    # Save predictions as pickle file
    predictions_dir = Path(config["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["model_id"].split("/")[-1]
    temp_tag = str(temperature).replace(".", "p")
    predictions_path = (
        predictions_dir
        / f"predictions_sampling_prob_{config['downstream_task']}_{model_name}_{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}_n{num_samples}_t{temp_tag}.pkl"
    )

    with open(predictions_path, "wb") as f:
        pickle.dump(subject_dicts, f)
