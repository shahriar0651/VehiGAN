from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_discriminator(cfg, model_cfg, wgan, dataset_dict, dis_file_name) -> dict:

    eval_dict = {}

    for attack in tqdm(dataset_dict.keys()):
        if attack == "No Attack": #FIXME: Remove condition
            continue
        x_test = dataset_dict[attack]['x_data']
        y_test = dataset_dict[attack]['y_data']
    
        # Load discriminator...
        discriminator = wgan.discriminator
        y_pred_attack = - discriminator.predict(x_test).flatten()
        y_test_attack = y_test
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
        output_file_path = Path(f"{dis_file_name}_{attack}_pred.csv")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        pred_data.to_csv(output_file_path)
    
        
    output_file_path = Path(f"{dis_file_name}_dict.json")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path,'w') as fp:
        fp.write(json.dumps(eval_dict, indent = 4))

    return eval_dict





