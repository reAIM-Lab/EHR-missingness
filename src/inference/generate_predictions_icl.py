import os
import json
import pickle
import random
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from openai import OpenAI

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
    extract_prediction,
    serialize_data,
)
from src.data.data_utils import sample_df

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

def _format_icl_examples(
    subject_dicts_train, num_examples, rng, stratified=True, pool_size=None
):
    subject_ids = list(subject_dicts_train.keys())
    if not subject_ids:
        return ""

    if pool_size is None:
        pool_size = num_examples

    if stratified:
        sampled_ids = _sample_stratified_ids(subject_dicts_train, pool_size, rng)
    else:
        k = min(pool_size, len(subject_ids))
        sampled_ids = rng.sample(subject_ids, k)

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
        prompt += "\n\nHere are labeled examples from this hospital:\n"
        prompt += examples_block
        prompt += "\n\nNow output response for the next patient.\n"
    prompt += "\n### Patient\n"
    prompt += target_serialization
    prompt += "\n### Response:\n"
    return prompt


def generate_predictions_icl(config):
    # Load in preprocessed MEDS data
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

    subject_dict_path = (
        Path(config["target_dir"])
        / f"{config['downstream_task']}_subject_dict_{config['explicit_missingness']}_{config['labs_only']}.pkl"
    )

    if subject_dict_path.exists():
        with open(subject_dict_path, "rb") as f:
            data_dicts = pickle.load(f)
        print(f"Loaded subject_dicts from {subject_dict_path}")
    else:
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

    query = get_detailed_instruct(config)
    subject_dicts = data_dicts.get("eval", data_dicts)
    unique_subjects = list(subject_dicts.keys())

    rng = random.Random(config.get("data_seed", 0))
    icl_num_examples = int(config.get("icl_num_examples", 20))
    icl_pool_size = int(config.get("icl_pool_size", icl_num_examples))
    stratified_icl = bool(config.get("icl_stratified", True))
    examples_block = _format_icl_examples(
        data_dicts["train"],
        icl_num_examples,
        rng,
        stratified=stratified_icl,
        pool_size=icl_pool_size,
    )

    input_texts = [
        _build_icl_prompt(query, examples_block, subject_dicts[subject_id]["serialization"])
        for subject_id in unique_subjects
    ]

    cache_dir = config.get("cache_dir")
    model_name = config["model_id"]
    predictions_dir = Path(config["predictions_dir"])
    icl_tag = f"icl{icl_num_examples}"
    prediction_basename = (
        f"predictions_{config['downstream_task']}_{config['model_id'].split('/')[-1]}_"
        f"{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}_{icl_tag}"
    )
    if os.path.exists(predictions_dir / f"{prediction_basename}.pkl"):
        print(model_name, "ALREADY RUN")
        return True

    if model_name == "openai/gpt-5":
        os.environ["OPENAI_API_KEY"] = config["key"]
        id = config.get("id", None)
        client = OpenAI()

        query_files = "batch_input.jsonl"

        # If no id - create query and save
        if id is None:
            # Save file
            with open(query_files, "w", encoding="utf-8") as f:
                for i, prompt in enumerate(input_texts):
                    request_obj = {
                        "custom_id": f"{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-5",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_completion_tokens": 4000,
                        },
                    }
                    f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

            print(f"Saved {len(input_texts)} requests")

            # Submit 24 hours query
            batch_input_file = client.files.create(
                file=open(query_files, "rb"),
                purpose="batch",
            )

            batch = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",  # maximum allowed time
            )

            print("Batch submitted")
            print("-" * 42)
            print("Batch ID:", batch.id)

            return None
        else:
            # Query server for query
            response = client.batches.retrieve(id)
            print(response)
            status = response.status
            print(f"Status: {status}")

            if status == "completed":
                print("Batch completed!")
            else:
                print("Batch unfinished!")
                return None

            # Process
            response = client.files.content(response.output_file_id).text
            resp_map = {}
            for line in response.splitlines():
                if line.strip():
                    obj = json.loads(line)
                    cid = int(obj["custom_id"])
                    body = obj["response"]["body"]
                    resp_map[cid] = body["choices"][0]["message"]["content"]

            outputs = [resp_map[cid] for cid in sorted(resp_map)]

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
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20000, stop_token_ids=stop_token_ids)

        # Initialize the vLLM engine
        llm = LLM(model=model_name, 
                  max_model_len=60000, 
                  tensor_parallel_size=1,
                  gpu_memory_utilization=0.9,
                  download_dir=cache_dir,)

        outputs = llm.generate(
            prompts=inputs,
            sampling_params=sampling_params,
        )

    elif model_name.startswith("Qwen/"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cache_dir)
        llm = LLM(
            model=model_name,
            max_model_len=30000,
            tensor_parallel_size=1,
            download_dir=cache_dir,
        )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=15000)

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
        if model_name.startswith("mistralai/"):
            llm = LLM(
                model=model_name,
                max_model_len=6000,
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
                download_dir=cache_dir,
            )
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ]
                for prompt in input_texts
            ]

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cache_dir)
            llm = LLM(
                model=model_name,
                max_model_len=30000,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                download_dir=cache_dir,
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

        sampling_params = SamplingParams(temperature=0.0, max_tokens=15000)

        if model_name.startswith("mistralai/"):
            outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        else:
            outputs = llm.generate(texts, sampling_params)

    # Collect results
    for i, output_text in enumerate(outputs):
        if model_name != "openai/gpt-5":
            output_text = output_text.outputs[0].text
        pred = extract_prediction(output_text)

        subject_id = unique_subjects[i]
        subject_dicts[subject_id]["response"] = output_text
        subject_dicts[subject_id]["prediction"] = pred

    # Save predictions as pickle file
    predictions_dir = Path(config["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_short_name = config["model_id"].split("/")[-1]

    predictions_path = (
        predictions_dir
        / f"predictions_{config['downstream_task']}_{model_short_name}_"
        f"{config['explicit_missingness']}_{config['labs_only']}_{config['include_cot_prompt']}_{icl_tag}.pkl"
    )

    with open(predictions_path, "wb") as f:
        pickle.dump(subject_dicts, f)
