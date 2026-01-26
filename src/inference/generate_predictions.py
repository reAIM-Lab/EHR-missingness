import os 
import numpy as np
import polars as pl
import pandas as pd
import torch
import json
import pickle
from pathlib import Path
from transformers import AutoTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import KFold
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TokensPrompt 
from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
    
from src.inference.inference_utils import get_detailed_instruct, extract_prediction, serialize_data, extract_structured_data
from src.data.data_utils import preprocess_df, sample_df

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def generate_predictions(config):
    # Load in preprocessed MEDS data
    new_meds_dir = Path(config['target_dir']) / f"{config['downstream_task']}_MEDS"
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

    subject_dict_path = Path(config['target_dir']) / f"{config['downstream_task']}_subject_dict_{config['explicit_missingness']}_{config['labs_only']}.pkl"

    if subject_dict_path.exists():
        with open(subject_dict_path, 'rb') as f:
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
                "label": label
            }
        
        data_dicts["eval"] = subject_dicts_test

        subject_groups = train_data.partition_by("subject_id", as_dict=True)
        subject_dicts_train = {}
        for subject_id, subject_data in subject_groups.items():
            serialization = serialize_data(subject_data, config)
            label = subject_data["boolean_value"][0]
            subject_dicts_train[subject_id[0]] = {
                "serialization": serialization,
                "label": label
            }
        
        data_dicts["train"] = subject_dicts_train

        # Save subject_dicts to target directory
        with open(subject_dict_path, 'wb') as f:
            pickle.dump(data_dicts, f)
        
        print(f"Saved subject_dicts to {subject_dict_path}")

    query = get_detailed_instruct(config)
    # subject_dicts = data_dicts['eval']
    subject_dicts = data_dicts.get('eval', data_dicts)
    unique_subjects = list(subject_dicts.keys())
    input_texts = [query + subject_dicts[subject_id]["serialization"] for subject_id in unique_subjects]
    
    model_name = config['model_id']
    predictions_dir = Path(config['predictions_dir'])
    if os.path.exists(predictions_dir / f'predictions_{config["downstream_task"]}_{config['model_id'].split("/")[-1]}_{config["explicit_missingness"]}_{config["labs_only"]}_{config["include_cot_prompt"]}.pkl'):
        print(model_name, "ALREADY RUN")
        return True

    if model_name == "openai/gpt-5":
        os.environ['OPENAI_API_KEY'] = config['key']
        id = config.get('id', None)
        client = OpenAI()

        query_files = 'batch_input.jsonl'

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
                            "max_completion_tokens": 4000
                        }
                    }
                    f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

            print(f"Saved {len(input_texts)} requests") 

            # Submit 24 hours query
            batch_input_file = client.files.create(
                file=open(query_files, "rb"),
                purpose="batch"
            )

            batch = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",   # maximum allowed time
            )

            print("Batch submitted")
            print('-' * 42)
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
            ## Reorder response
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
            ) for prompt in input_texts
        ]
        
        prefill_ids = [
            encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            for convo in convos
        ]
        inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prefill_ids]
        
        # Harmony stop tokens (pass to sampler so they won't be included in output)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4000, stop_token_ids=stop_token_ids)

        # Initialize the vLLM engine
        llm = LLM(model=model_name, max_model_len=6000, tensor_parallel_size = 2)

        outputs = llm.generate(
            prompts=inputs,
            sampling_params=sampling_params
            )

    elif model_name.startswith("Qwen/"):
        if config['use_finetuned']:
            if config['finetune_method'] == 'grpo':
                adapter_dir = f"{model_name}-GRPO/checkpoint-500" 
            elif config['finetune_method'] == 'dpo':
                adapter_dir = f"{model_name}-DPO/checkpoint-1455" 
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir, padding_side="left")
            llm = LLM(model=model_name, 
                      max_model_len=6000,
                      enable_lora=True
                      )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left") 
            llm = LLM(
                model=model_name,
                max_model_len=6000,
                tensor_parallel_size = 2
            )
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4000)

        messages = [
            {"role": "user", "content": prompt} for prompt in input_texts
        ]
        texts = [
            tokenizer.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message in messages
        ]

        if config['use_finetuned']:
            if config['finetune_method'] == 'grpo':
                outputs = llm.generate(texts, sampling_params, lora_request=LoRARequest("grpo_adapter", 1, adapter_dir))
            elif config['finetune_method'] == 'dpo':
                outputs = llm.generate(texts, sampling_params, lora_request=LoRARequest("dpo_adapter", 1, adapter_dir))
        else:
            outputs = llm.generate(texts, sampling_params)
    
    else:
        if model_name.startswith("mistralai/"):
            llm = LLM(
                model=model_name,
                max_model_len=6000,
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
            )
            messages = [
                [
                    {"role": "user", "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        }
                        ]
                    }
                ] for prompt in input_texts
            ]

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left") 
            llm = LLM(
                model=model_name,
                max_model_len=6000,
                tensor_parallel_size = 2
            )

            messages = [
                {"role": "user", "content": prompt} for prompt in input_texts
            ]

            texts = [
                tokenizer.apply_chat_template(
                    [message],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for message in messages
            ]
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4000)

        if model_name.startswith("mistralai/"):
            outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        else:
            outputs = llm.generate(texts, sampling_params)

    # Collect results
    for i, output_text in enumerate(outputs):
        if model_name != 'openai/gpt-5':
            output_text = output_text.outputs[0].text
        pred = extract_prediction(output_text)

        subject_id = unique_subjects[i]
        subject_dicts[subject_id]["response"] = output_text
        subject_dicts[subject_id]["prediction"] = pred

    # Save predictions as pickle file
    predictions_dir = Path(config['predictions_dir'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_name = config['model_id'].split("/")[-1]

    if config['use_finetuned']:
        if config['finetune_method'] == 'grpo':
            predictions_path = predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_{config["explicit_missingness"]}_{config["labs_only"]}_{config["include_cot_prompt"]}_{config["use_finetuned"]}_grpo.pkl'
        elif config['finetune_method'] == 'dpo':
            predictions_path = predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_{config["explicit_missingness"]}_{config["labs_only"]}_{config["include_cot_prompt"]}_{config["use_finetuned"]}_dpo.pkl'
    else:
        #predictions_path = predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_{config["explicit_missingness"]}_{config["labs_only"]}_{config["include_cot_prompt"]}_{config["use_finetuned"]}.pkl'
        predictions_path = predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_{config["explicit_missingness"]}_{config["labs_only"]}_{config["include_cot_prompt"]}.pkl'

    with open(predictions_path, 'wb') as f:
        pickle.dump(subject_dicts, f)

def generate_baseline_predictions(config):
    # Load in preprocessed MEDS data
    new_meds_dir = Path(config['target_dir']) / f"{config['downstream_task']}_MEDS"
    train_files = sorted((new_meds_dir / "data" / "train").glob("*.parquet"))
    tune_files = sorted((new_meds_dir / "data" / "tuning").glob("*.parquet"))
    test_files = sorted((new_meds_dir / "data" / "held_out").glob("*.parquet"))

    processed_path = Path(config['target_dir']) / f"{config['downstream_task']}_structured_df.csv"
    if processed_path.exists():
        features_df = pd.read_csv(processed_path)
    else:
        # Load and concatenate all data
        dfs = []
        for files in [train_files, tune_files, test_files]:
            for file in files:
                df = pl.read_parquet(file)
                dfs.append(df)
        
        data = pl.concat(dfs)
        test_data, _ = sample_df(data, config)
        subject_groups = test_data.partition_by("subject_id", as_dict=True)

        all_features = []
        for subject_id, subject_data in subject_groups.items():
            label = subject_data["boolean_value"][0]
            feature_dict = extract_structured_data(subject_data, config)
            feature_dict['subject_id'] = subject_id[0]  # Add subject_id to the feature dict
            feature_dict['label'] = label
            all_features.append(feature_dict)
        
        # Create dataframe with all features
        features_df = pd.DataFrame(all_features)
        if 'race' in features_df.columns:
            features_df['race'] = features_df['race'].map(lambda x: x if x in ['unknown', 'white'] else 'non-white')
        features_df = preprocess_df(features_df)
        features_df.to_csv(processed_path, index=False)

    # Drop subject_id and split columns from both dataframes
    subject_ids = features_df['subject_id']
    features_df = features_df.drop(columns=['subject_id'])

    # Compute missingness rate for each feature
    print("\nMissingness rates for each feature:")
    feature_cols = features_df.drop(columns=['label']).columns
    for col in feature_cols:
        missing_rate = features_df[col].isnull().mean()
        print(f"{col}: {missing_rate:.3f} ({missing_rate*100:.1f}%)")
    
    # Overall missingness statistics
    total_missing = features_df.drop(columns=['label']).isnull().sum().sum()
    total_values = features_df.drop(columns=['label']).size
    overall_missing_rate = total_missing / total_values
    print(f"\nOverall missingness rate: {overall_missing_rate:.3f} ({overall_missing_rate*100:.1f}%)")

    # Prepare features and labels
    X = features_df.drop(columns=['label']).fillna(0)
    y = features_df['label']

    label_counts = y.value_counts(normalize=True)  # gives proportion per class
    positive_prevalence = label_counts.get(1, 0)   # prevalence of label=1

    print(f"Label prevalence (positive class): {positive_prevalence:.3f} ({positive_prevalence*100:.1f}%)")
    print("\nFull label distribution:")
    print(y.value_counts(normalize=True).rename("proportion"))

    # Create missingness indicators
    X_with_missing = features_df.drop(columns=['label']).copy()
    
    # Add missingness indicator columns
    for col in X_with_missing.columns:
        missing_frac = X_with_missing[col].isnull().mean()
        if missing_frac > 0:  # >0% missing values
            X_with_missing[f'{col}_missing'] = X_with_missing[col].isnull().astype(int)

    # Calculate number of missing features per patient
    missing_cols = [c for c in X_with_missing.columns if c.endswith("_missing")]
    missing_counts = X_with_missing[missing_cols].sum(axis=1)
    print(f"\nMissing features per patient - Mean: {missing_counts.mean():.2f}, Std: {missing_counts.std():.2f}")
    print(f"Min missing features: {missing_counts.min()}, Max missing features: {missing_counts.max()}")
    
    # Fill missing values with 0
    X_with_missing = X_with_missing.fillna(0)
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Storage for out-of-fold predictions
    y_pred_no_missing = np.zeros(len(y))
    y_pred_with_missing = np.zeros(len(y))
    coefs_with_missing = []

    for train_idx, val_idx in kf.split(X):
        # ----- Model 1: without missingness indicators -----
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        
        lr_no_missing = LogisticRegression(random_state=42, max_iter=1000)
        lr_no_missing.fit(X_train, y_train)
        y_pred_no_missing[val_idx] = lr_no_missing.predict_proba(X_val)[:, 1]

        # ----- Model 2: with missingness indicators -----
        X_train_miss = X_with_missing.iloc[train_idx]
        X_val_miss = X_with_missing.iloc[val_idx]
        y_train = y.iloc[train_idx]  # same labels

        lr_with_missing = LogisticRegression(random_state=42, max_iter=1000)
        lr_with_missing.fit(X_train_miss, y_train)
        y_pred_with_missing[val_idx] = lr_with_missing.predict_proba(X_val_miss)[:, 1]
        coefs_with_missing.append(lr_with_missing.coef_[0])

    coefs_with_missing = np.vstack(coefs_with_missing)
    avg_coefs_with_missing = coefs_with_missing.mean(axis=0)

    # Get feature importances for model without missingness indicators
    feature_importances_with_missing = dict(zip(X_with_missing.columns, avg_coefs_with_missing))
    
    # Print all feature importances for model with missingness indicators
    sorted_all_importances = sorted(feature_importances_with_missing.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
    
    # Evaluate both models
    auc_no_missing = roc_auc_score(y, y_pred_no_missing)
    auc_with_missing = roc_auc_score(y, y_pred_with_missing)
    
    logloss_no_missing = log_loss(y, y_pred_no_missing)
    logloss_with_missing = log_loss(y, y_pred_with_missing)
    
    print(f"Logistic Regression without missingness indicators - AUC: {auc_no_missing:.4f}, Log Loss: {logloss_no_missing:.4f}")
    print(f"Logistic Regression with missingness indicators - AUC: {auc_with_missing:.4f}, Log Loss: {logloss_with_missing:.4f}")
    
    # Save baseline predictions
    baseline_predictions = {
        'lr_no_missing': {
            'subject_ids': subject_ids,
            'predictions': y_pred_no_missing,
            'labels': y.values,
            'auc': auc_no_missing,
            'log_loss': logloss_no_missing
        },
        'lr_with_missing': {
            'subject_ids': subject_ids,
            'predictions': y_pred_with_missing,
            'labels': y.values,
            'auc': auc_with_missing,
            'log_loss': logloss_with_missing,
            'feature_importances': sorted_all_importances,
            'missing_counts': missing_counts.values
        }
    }
    
    predictions_dir = Path(config['predictions_dir'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    with open(predictions_dir / f'baseline_predictions_{config["downstream_task"]}.pkl', 'wb') as f:
        pickle.dump(baseline_predictions, f)



