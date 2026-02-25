import argparse
import yaml
from pathlib import Path

from src.inference.generate_predictions_icl import generate_predictions_icl

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ehrLLM experiments")
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--config', type=str, default="./configs", help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['test_icl'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(Path(args.config) / f"{args.experiment}.yaml") if args.config else {}

    if args.mode == 'test_icl':
        base_model_id = str(config.get('model_id', '')).lower()
        if 'qwen' in base_model_id:
            model_ids = [
                "Qwen/Qwen3-4B",
                "Qwen/Qwen3-8B",
                "Qwen/Qwen3-14B",
                "Qwen/Qwen3-32B",
            ]
            seed_count = 5
            num_generations = 5
        elif 'gemma' in base_model_id:
            model_ids = [
                "google/medgemma-27b-text-it",
                "google/gemma-3-27b-it",
            ]
            seed_count = 5
            num_generations = 5
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
            for missingness in [True]:
                config['explicit_missingness'] = missingness
                for icl_samples in [0]:
                    config['icl_num_examples'] = icl_samples
                    for include_cot_prompt in [True]:
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
    