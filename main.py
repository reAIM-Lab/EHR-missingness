import argparse
import yaml
from pathlib import Path

from src.data.generate_cohort import generate_cohort_cuimc, generate_cohort_mimic
from src.inference.generate_predictions import generate_predictions, generate_baseline_predictions
from src.inference.evaluate import evaluate_predictions

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ehrLLM experiments")
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name', choices=['mimic_los', 'mimic_death', 'mimic_readmission', 'cuimc_los', 'cuimc_readmission', 'cuimc_death'])
    parser.add_argument('--config', type=str, default="./configs", help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['generate_cohort', 'test', 'baseline', 'evaluate'])
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(Path(args.config) / f"{args.experiment}.yaml") if args.config else {}

    if args.mode == 'generate_cohort':
        if 'cuimc' in args.experiment:
            generate_cohort_cuimc(config) # Filter MEDS input to generate downstream evaluation cohort
        elif 'mimic' in args.experiment:
            generate_cohort_mimic(config) # Filter MEDS input to generate downstream evaluation cohort
    elif args.mode == 'test':
        generate_predictions(config) # Run inference on downstream evaluation cohort 
    elif args.mode == 'baseline':
        generate_baseline_predictions(config)
    elif args.mode == 'evaluate':
        evaluate_predictions(config) # Evaluate predictions

if __name__ == "__main__":
    main()