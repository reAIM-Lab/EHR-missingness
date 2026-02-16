import argparse
import yaml
from pathlib import Path

from src.data.generate_cohort import generate_cohort_mimic
from src.inference.generate_predictions import generate_predictions, generate_baseline_predictions
from src.inference.generate_predictions_icl import generate_predictions_icl
from src.inference.generate_predictions_sampling import generate_predictions_sampling, generate_predictions_sampling_probability

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ehrLLM experiments")
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--config', type=str, default="./configs", help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['generate_cohort', 'test', 'test_icl', 'iteration', 'consistency', 'baseline'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(Path(args.config) / f"{args.experiment}.yaml") if args.config else {}

    if args.mode == 'generate_cohort':
        generate_cohort_mimic(config) # Filter MEDS input to generate downstream evaluation cohort
    elif args.mode == 'baseline':
        generate_baseline_predictions(config)
    elif args.mode == 'test_icl':
        base_model_id = str(config.get('model_id', '')).lower()
        if 'qwen' in base_model_id:
            model_ids = [
                # "Qwen/Qwen3-1.7B",
                #"Qwen/Qwen3-4B",
                #"Qwen/Qwen3-8B",
                #"Qwen/Qwen3-14B",
                "Qwen/Qwen3-32B",
            ]
            seed_count = 5
            num_generations = 5
        elif 'anthropic' in base_model_id or 'claude' in base_model_id or 'gemini' in base_model_id:
            model_ids = [config['model_id']]
            seed_count = 3
            num_generations = 3
        else:
            model_ids = [config['model_id']]
            seed_count = 5
            num_generations = 5

        config['num_generations'] = num_generations
        print(
            f"[test_icl] model_family={base_model_id}, models={model_ids}, "
            f"seeds=1..{seed_count}, num_generations={num_generations}"
        )

        for model in model_ids:
            config['model_id'] = model
            for missingness in [True, False]:
                config['explicit_missingness'] = missingness
                for icl_samples in [0, 20, 50]:
                    config['icl_num_examples'] = icl_samples
                    for include_cot_prompt in [True, False]:
                        if (not missingness) and include_cot_prompt:
                            continue

                        config['include_cot_prompt'] = include_cot_prompt

                        seeds = list(range(1, seed_count + 1)) if icl_samples > 0 else [1]
                        for data_seed in seeds:
                            config['data_seed'] = data_seed
                            try:
                                generate_predictions_icl(config)
                            except Exception as e:
                                print("Error with", model)
                                print(e)
                            
    elif args.mode == 'iteration':
        for model in ["openai/gpt-oss-120b", "google/medgemma-27b-text-it"]:
            config['model_id'] = model
            for explicit_missingness in [True, False]:
                config['explicit_missingness'] = explicit_missingness
                for include_cot_prompt in [True, False]:
                    if (not explicit_missingness) and include_cot_prompt:
                        continue

                    config['include_cot_prompt'] = include_cot_prompt
                    try:
                        generate_predictions_sampling(config)
                    except Exception as e:
                        print("Error with", model)
                        print(e)
    elif args.mode == 'consistency':
        for model in ["openai/gpt-oss-120b", "google/medgemma-27b-text-it"]:
            config['model_id'] = model
            for explicit_missingness in [True, False]:
                config['explicit_missingness'] = explicit_missingness
                for include_cot_prompt in [True, False]:
                    if (not explicit_missingness) and include_cot_prompt:
                        continue

                    config['include_cot_prompt'] = include_cot_prompt
                    try:
                        generate_predictions_sampling_probability(config)
                    except Exception as e:
                        print("Error with", model)
                        print(e)
    else:
        for model in ["google/gemma-3-4b-it", "google/gemma-3-27b-it", "google/medgemma-27b-text-it",
                    "openai/gpt-oss-20b", "openai/gpt-oss-120b",
                    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
                    "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"]:
            config['model_id'] = model
            for explicit_missingness in [True, False]:
                config['explicit_missingness'] = explicit_missingness
                for include_cot_prompt in [True, False]:
                    config['include_cot_prompt'] = include_cot_prompt
                    try:
                        generate_predictions(config)
                    except Exception as e:
                        print("Error with", model)
                        print(e)