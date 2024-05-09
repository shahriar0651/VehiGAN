from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import time


def test_baseline(cfg, model_name, model, dataset_dict, baseline_file_name):
    eval_dict = {}
    for attack, dataset in dataset_dict.items():
        df_prediction = dataset[["attack_name", "attack_gt"]]
        x_test_attack = dataset[cfg["features"]].values
        y_test_attack = dataset["attack_gt"].values
        y_pred_attack = model.decision_function(x_test_attack)

        df_prediction[f"pred_score"] =  y_pred_attack 
        y_test_attack[y_test_attack > 0] = 1
        try:
            auroc_score = roc_auc_score(y_test_attack, y_pred_attack)
            auprc_score = average_precision_score(y_test_attack, y_pred_attack)

            eval_dict_attack =  {
                "Attack": attack,        
                "AUROC": auroc_score, 
                "AUPRC": auprc_score
                }
            eval_dict[attack] = eval_dict_attack
        except:
            print("No attack traces!!!")

        pred_data = pd.DataFrame()
        pred_data['prediction'] = y_pred_attack
        pred_data['ground_truth'] = y_test_attack
        output_file_path = Path(f"{baseline_file_name}_{attack}_pred.csv")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        pred_data.to_csv(output_file_path)
    
    output_file_path = Path(f"{baseline_file_name}_dict.json")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path,'w') as fp:
        fp.write(json.dumps(eval_dict, indent = 4))

