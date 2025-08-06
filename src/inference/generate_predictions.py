import polars as pl
import torch
import pickle
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.inference.inference_utils import get_detailed_instruct, extract_prediction, serialize_data

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