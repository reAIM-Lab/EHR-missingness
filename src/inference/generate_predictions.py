import polars as pl
import pandas as pd
import torch
import pickle
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
    
from src.inference.inference_utils import get_detailed_instruct, extract_prediction, serialize_data, extract_structured_data
from src.data.data_utils import preprocess_df

def generate_predictions(config):
    # Load in preprocessed MEDS data
    new_meds_dir = Path(config['target_dir']) / f"{config['downstream_task']}_MEDS"
    train_files = sorted((new_meds_dir / "data" / "train").glob("*.parquet"))
    tune_files = sorted((new_meds_dir / "data" / "tuning").glob("*.parquet"))
    test_files = sorted((new_meds_dir / "data" / "held_out").glob("*.parquet"))

    # Load and concatenate all data
    dfs = []
    #for files in [train_files, tune_files, test_files]:
    for files in [test_files]:
        for file in files:
            df = pl.read_parquet(file)
            dfs.append(df)
    
    data = pl.concat(dfs)

    subject_groups = data.partition_by("subject_id", as_dict=True)
    subject_dicts = {}
    for subject_id, subject_data in subject_groups.items():
        serialization = serialize_data(subject_data, config)
        label = subject_data["boolean_value"][0]
        subject_dicts[subject_id[0]] = {
            "serialization": serialization,
            "label": label
        }

    model_name = config['model_id']
    if model_name == "openai/gpt-oss-20b":
        # TODO: Format input for openAI models
        pass
    else:
        query = get_detailed_instruct(config)
        unique_subjects = list(subject_dicts.keys())
        input_texts = [query + subject_dicts[subject_id]["serialization"] for subject_id in unique_subjects]

        tokenizer = AutoTokenizer.from_pretrained(model_name) 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
        messages = [
            {"role": "user", "content": prompt} for prompt in input_texts
        ]
        texts = [
            tokenizer.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for message in messages
        ]
        tokenizer.padding_side = "left"
        batch_dict = tokenizer(texts, max_length=config['max_length'], padding=True, truncation=True, return_tensors='pt')

        dataset = TensorDataset(batch_dict['input_ids'], batch_dict['attention_mask'], torch.tensor(unique_subjects, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating predictions"):
                input_ids, attention_mask, subject_ids = [b.to(device) for b in batch]

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2000,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    # do_sample=True,
                    # temperature=0.7,
                    # top_p=0.8,
                    # top_k=20
                )

                for i in range(generated_ids.size(0)):
                    prompt_len = input_ids.size(1)
                    generated_only = generated_ids[i][prompt_len:]

                    output_text = tokenizer.decode(generated_only, skip_special_tokens=True).strip("\n")
                    pred = extract_prediction(output_text)
                    subject_id = subject_ids[i].item()
                    subject_dicts[subject_id]["response"] = output_text
                    subject_dicts[subject_id]["prediction"] = pred

    # Save predictions as pickle file
    predictions_dir = Path(config['predictions_dir'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_name = config['model_id'].split("/")[-1]
    with open(predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_{config["explicit_missingness"]}.pkl', 'wb') as f:
        pickle.dump(subject_dicts, f)

def generate_baseline_predictions(config):
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

    subject_groups = data.partition_by("subject_id", as_dict=True)
    all_features = []
    for subject_id, subject_data in subject_groups.items():
        label = subject_data["boolean_value"][0]
        feature_dict = extract_structured_data(subject_data, config)
        feature_dict['subject_id'] = subject_id[0]  # Add subject_id to the feature dict
        feature_dict['label'] = label
        feature_dict['split'] = subject_data['split'][0]
        all_features.append(feature_dict)
    
    # Create dataframe with all features
    features_df = pd.DataFrame(all_features)
    features_df['race'] = features_df['race'].map(lambda x: x if x in ['unknown', 'white'] else 'non-white')
    features_df = preprocess_df(features_df)

    train_df = features_df[features_df['split'].isin(['train', 'tuning'])]
    test_df = features_df[features_df['split'] == 'held_out']

    # Drop subject_id and split columns from both dataframes
    train_df = train_df.drop(columns=['subject_id', 'split'])
    test_df = test_df.drop(columns=['subject_id', 'split'])

    # Prepare features and labels
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    
    # Model 1: Random Forest without missingness indicators
    X_train_no_missing = X_train.fillna(0)  # Fill missing values with 0
    X_test_no_missing = X_test.fillna(0)
    
    rf_no_missing = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_no_missing.fit(X_train_no_missing, y_train)
    
    # Predictions for model without missingness indicators
    y_pred_no_missing = rf_no_missing.predict_proba(X_test_no_missing)[:, 1]
    
    # Model 2: Random Forest with missingness indicators
    # Create missingness indicators
    X_train_with_missing = X_train.copy()
    X_test_with_missing = X_test.copy()
    
    # Add missingness indicator columns
    for col in X_train.columns:
        if X_train[col].isnull().any():
            X_train_with_missing[f'{col}_missing'] = X_train[col].isnull().astype(int)
            X_test_with_missing[f'{col}_missing'] = X_test[col].isnull().astype(int)
    
    # Fill missing values with 0
    X_train_with_missing = X_train_with_missing.fillna(0)
    X_test_with_missing = X_test_with_missing.fillna(0)
    
    rf_with_missing = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_with_missing.fit(X_train_with_missing, y_train)
    
    # Predictions for model with missingness indicators
    y_pred_with_missing = rf_with_missing.predict_proba(X_test_with_missing)[:, 1]
    
    # Evaluate both models
    auc_no_missing = roc_auc_score(y_test, y_pred_no_missing)
    auc_with_missing = roc_auc_score(y_test, y_pred_with_missing)
    
    logloss_no_missing = log_loss(y_test, y_pred_no_missing)
    logloss_with_missing = log_loss(y_test, y_pred_with_missing)
    
    print(f"Random Forest without missingness indicators - AUC: {auc_no_missing:.4f}, Log Loss: {logloss_no_missing:.4f}")
    print(f"Random Forest with missingness indicators - AUC: {auc_with_missing:.4f}, Log Loss: {logloss_with_missing:.4f}")
    
    # Save baseline predictions
    baseline_predictions = {
        'rf_no_missing': {
            'predictions': y_pred_no_missing,
            'labels': y_test.values,
            'auc': auc_no_missing,
            'log_loss': logloss_no_missing
        },
        'rf_with_missing': {
            'predictions': y_pred_with_missing,
            'labels': y_test.values,
            'auc': auc_with_missing,
            'log_loss': logloss_with_missing
        }
    }
    
    predictions_dir = Path(config['predictions_dir'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    with open(predictions_dir / f'baseline_predictions_{config["downstream_task"]}.pkl', 'wb') as f:
        pickle.dump(baseline_predictions, f)

    # Get label prevalence
    # train_prevalence = train_df['label'].mean()
    # test_prevalence = test_df['label'].mean()
    
    # print(f"Training set label prevalence: {train_prevalence:.4f}")
    # print(f"Test set label prevalence: {test_prevalence:.4f}")



