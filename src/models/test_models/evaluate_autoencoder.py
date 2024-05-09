from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_autoencoder(cfg, model_cfg, autoencoder, dataset_dict, ae_file_name) -> dict:

    eval_dict = {}

    for attack in tqdm(dataset_dict.keys()):
        if attack == "No Attack":
            continue
        x_test = dataset_dict[attack]['x_data']
        y_test = dataset_dict[attack]['y_data']
    
        # Load autoencoder...
        x_test_recon = autoencoder.predict(x_test)
        y_pred_attack = np.linalg.norm(x_test_recon - x_test, axis=(1, 2)).flatten()
        print(y_pred_attack.shape, y_pred_attack[0:10])
        y_test_attack = y_test
        y_test_attack[y_test_attack > 0] = 1
        print(y_test_attack.shape, y_test_attack[0:10])

        try:
            auroc_score = roc_auc_score(y_test_attack, y_pred_attack)
            auprc_score = average_precision_score(y_test_attack, y_pred_attack)

            eval_dict_attack =  {
                "Attack": attack,        
                "AUROC": auroc_score, 
                "AUPRC": auprc_score
                }
            eval_dict[attack] = eval_dict_attack
            print(eval_dict_attack)
        except Exception as e: 
            print(e)
            print("No attack traces!!!")

        pred_data = pd.DataFrame()
        pred_data['prediction'] = y_pred_attack
        pred_data['ground_truth'] = y_test_attack
        output_file_path = Path(f"{ae_file_name}_{attack}_pred.csv")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        pred_data.to_csv(output_file_path)
    
        
    output_file_path = Path(f"{ae_file_name}_dict.json")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path,'w') as fp:
        fp.write(json.dumps(eval_dict, indent = 4))

    return eval_dict





