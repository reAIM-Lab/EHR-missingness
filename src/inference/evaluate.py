import pickle
from pathlib import Path

from sklearn.metrics import log_loss, roc_auc_score

def evaluate_predictions(config):
    predictions_dir = Path(config['predictions_dir'])
    model_name = config['model_id'].split("/")[-1]
    try:
        with open(predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_False_{config["labs_only"]}.pkl', 'rb') as f:
            subject_dicts_1 = pickle.load(f)
    except FileNotFoundError:
        print(f"No predictions found for {config['downstream_task']} with explicit missingness False")
        return None

    try:
        with open(predictions_dir / f'predictions_{config["downstream_task"]}_{model_name}_True_{config["labs_only"]}.pkl', 'rb') as f:
            subject_dicts_2 = pickle.load(f)
    except FileNotFoundError:
        print(f"No predictions found for {config['downstream_task']} with explicit missingness True")
        return None

    labels = [subject_dicts_1[subject_id]["label"] for subject_id in subject_dicts_1.keys()]
    predictions_1 = [subject_dicts_1[subject_id]["prediction"] for subject_id in subject_dicts_1.keys()]
    predictions_2 = [subject_dicts_2[subject_id]["prediction"] for subject_id in subject_dicts_2.keys()]

    # Print sample generations
    prompt = [subject_dicts_2[subject_id]["serialization"] for subject_id in subject_dicts_2.keys()]
    generations = [subject_dicts_2[subject_id]["response"] for subject_id in subject_dicts_2.keys()]

    print(prompt[1].replace('\\n', '\n'))
    print(generations[1].replace('\\n', '\n'))

    print(len(predictions_1))
    print(len(predictions_2))
    print(len(labels))
    
    # Compute metrics
    try:
        # Filter out None predictions, using same indices for both sets
        valid_indices = [i for i, (pred1, pred2) in enumerate(zip(predictions_1, predictions_2)) 
                        if pred1 is not None and pred2 is not None]
        
        filtered_labels = [labels[i] for i in valid_indices]
        filtered_preds_1 = [predictions_1[i] for i in valid_indices]
        filtered_preds_2 = [predictions_2[i] for i in valid_indices]

        logloss_1 = log_loss(filtered_labels, filtered_preds_1)
        logloss_2 = log_loss(filtered_labels, filtered_preds_2)
        auroc_1 = roc_auc_score(filtered_labels, filtered_preds_1)
        auroc_2 = roc_auc_score(filtered_labels, filtered_preds_2)
        
        print(f"No Missingness Log Loss: {logloss_1:.4f}")
        print(f"No Missingness AUROC: {auroc_1:.4f}")

        print(f"Missingness Log Loss: {logloss_2:.4f}")
        print(f"Missingness AUROC: {auroc_2:.4f}")
        
        # Return metrics dictionary
        metrics = {
            "log_loss_1": logloss_1,
            "auroc_1": auroc_1,
            "log_loss_2": logloss_2,
            "auroc_2": auroc_2
        }
        return metrics
        
    except Exception as e:
        print(f"Error computing metrics: {str(e)}")
        return None