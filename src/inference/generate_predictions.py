import polars as pl
import pandas as pd
import torch
import pickle
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
    
from src.inference.inference_utils import get_detailed_instruct, extract_prediction, serialize_data, extract_structured_data
from src.data.data_utils import preprocess_df, balance_dataset, sample_df

def generate_predictions(config):
    # Load in preprocessed MEDS data
    new_meds_dir = Path(config['target_dir']) / f"{config['downstream_task']}_MEDS"
    train_files = sorted((new_meds_dir / "data" / "train").glob("*.parquet"))
    tune_files = sorted((new_meds_dir / "data" / "tuning").glob("*.parquet"))
    test_files = sorted((new_meds_dir / "data" / "held_out").glob("*.parquet"))

    # Load and concatenate all data
    dfs = []
    for files in [train_files, tune_files, test_files]:
    #for files in [test_files]:
        for file in files:
            df = pl.read_parquet(file)
            dfs.append(df)
    
    data = pl.concat(dfs)
    #data = balance_dataset(data, config, return_prevalence=False)
    data = sample_df(data, config)

    subject_dict_path = Path(config['target_dir']) / f"{config['downstream_task']}_subject_dict_{config['explicit_missingness']}_{config['labs_only']}.pkl"

    if subject_dict_path.exists():
        with open(subject_dict_path, 'rb') as f:
            subject_dicts = pickle.load(f)
        print(f"Loaded subject_dicts from {subject_dict_path}")
    else:
        subject_groups = data.partition_by("subject_id", as_dict=True)
        subject_dicts = {}
        for subject_id, subject_data in subject_groups.items():
            serialization = serialize_data(subject_data, config)
            label = subject_data["boolean_value"][0]
            subject_dicts[subject_id[0]] = {
                "serialization": serialization,
                "label": label
            }

        # Save subject_dicts to target directory
        with open(subject_dict_path, 'wb') as f:
            pickle.dump(subject_dicts, f)
        
        print(f"Saved subject_dicts to {subject_dict_path}")

    query = get_detailed_instruct(config)
    unique_subjects = list(subject_dicts.keys())
    input_texts = [query + subject_dicts[subject_id]["serialization"] for subject_id in unique_subjects]

    model_name = config['model_id']
    if model_name == "openai/gpt-oss-20b":
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left") 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            attn_implementation="kernels-community/vllm-flash-attn3",
        )

        messages = [
            [{"role": "user", "content": prompt}] # each conversation is a list of messages 
        for prompt in input_texts]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True,
            ).to(model.device)
        
        input_ids = inputs["input_ids"]
        attention_masks = inputs["attention_mask"]

        dataset = TensorDataset(input_ids, attention_masks, torch.tensor(unique_subjects, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left") 
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
        batch_dict = tokenizer(texts, max_length=config['max_length'], padding=True, truncation=True, return_tensors='pt')

        dataset = TensorDataset(batch_dict['input_ids'], batch_dict['attention_mask'], torch.tensor(unique_subjects, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids, attention_mask, subject_ids = [b.to(device) for b in batch]

            if config['hidden_states']:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2000,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    return_dict_in_generate=True,   # return full outputs
                    output_hidden_states=True
                )
                generated_ids = outputs.sequences
            else:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2000,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
                outputs = None

            input_len = input_ids.size(1)
            gen_lengths = (generated_ids != tokenizer.pad_token_id).sum(dim=1)
            prompt_lens = (input_ids != tokenizer.pad_token_id).sum(dim=1)

            for i in range(generated_ids.size(0)):
                generated_only = generated_ids[i][input_len:]
                output_text = tokenizer.decode(generated_only, skip_special_tokens=True).strip("\n")
                pred = extract_prediction(output_text)
                subject_id = subject_ids[i].item()
                subject_dicts[subject_id]["response"] = output_text
                subject_dicts[subject_id]["prediction"] = pred

                if config['hidden_states']:
                    length = gen_lengths[i].item() - prompt_lens[i].item()
                    last_layer_hs = outputs.hidden_states[length-1][-1].cpu()
                    sample_hs = last_layer_hs[i].squeeze().to(torch.float32).numpy()
                    subject_dicts[subject_id]["hidden_states"] = sample_hs

            if outputs is not None:
                del outputs
                torch.cuda.empty_cache()

    # Save predictions as pickle file
    predictions_dir = Path(config['predictions_dir'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_name = config['model_id'].split("/")[-1]
    with open(predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_{config["explicit_missingness"]}_{config["labs_only"]}_{config['include_missingness_prompt']}.pkl', 'wb') as f:
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
        #for files in [test_files]:
            for file in files:
                df = pl.read_parquet(file)
                dfs.append(df)
        
        data = pl.concat(dfs)
        # data, pi = balance_dataset(data, config, return_prevalence=True)
        data = sample_df(data, config)
        subject_groups = data.partition_by("subject_id", as_dict=True)

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

        # with open(str(processed_path).replace(".csv", "_prevalence.txt"), "w") as f:
        #     f.write(str(pi))

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

    # Model 1: Random Forest without missingness indicators
    lr_no_missing = LogisticRegression(random_state=42)
    lr_no_missing.fit(X, y)
    
    # Predictions for model without missingness indicators
    y_pred_no_missing = lr_no_missing.predict_proba(X)[:, 1]
    
    # Create missingness indicators
    X_with_missing = features_df.drop(columns=['label']).copy()
    
    # Add missingness indicator columns
    for col in X_with_missing.columns:
        if X_with_missing[col].isnull().any():
            X_with_missing[f'{col}_missing'] = X_with_missing[col].isnull().astype(int)

    # Calculate number of missing features per patient
    missing_cols = [c for c in X_with_missing.columns if c.endswith("_missing")]
    missing_counts = X_with_missing[missing_cols].sum(axis=1)
    print(f"\nMissing features per patient - Mean: {missing_counts.mean():.2f}, Std: {missing_counts.std():.2f}")
    print(f"Min missing features: {missing_counts.min()}, Max missing features: {missing_counts.max()}")
    
    # Fill missing values with 0
    X_with_missing = X_with_missing.fillna(0)
    
    lr_with_missing = LogisticRegression(random_state=42)
    lr_with_missing.fit(X_with_missing, y)

    # Get feature importances for model without missingness indicators
    feature_names_with_missing = X_with_missing.columns.tolist()
    feature_importances_with_missing = dict(zip(feature_names_with_missing, lr_with_missing.coef_[0]))
    
    # Print all feature importances for model with missingness indicators
    sorted_all_importances = sorted(feature_importances_with_missing.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
    
    # print("\nAll Feature Importances (Logistic Regression with missingness indicators):")
    # for feature, importance in sorted_all_importances:
    #     print(f"{feature}: {importance:.4f}")
    
    # Filter to only missingness indicator features
    # missingness_importances = {k: v for k, v in feature_importances_with_missing.items() if k.endswith('_missing')}
    # sorted_missingness_importances = sorted(missingness_importances.items(), 
    #                                       key=lambda x: abs(x[1]), reverse=True)
    # print("\nMissingness Indicator Feature Importances (Logistic Regression with missingness indicators):")
    # for feature, importance in sorted_missingness_importances:
    #     print(f"{feature}: {importance:.4f}")
    
    # Predictions for model with missingness indicators
    y_pred_with_missing = lr_with_missing.predict_proba(X_with_missing)[:, 1]
    
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



