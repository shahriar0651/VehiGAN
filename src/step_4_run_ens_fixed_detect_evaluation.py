import os
import hydra
from pathlib import Path
from omegaconf import DictConfig
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dataset import *
from models import *
from helper import *

@hydra.main(config_path="../config", config_name="config.yaml")
def run_ens_fixed_detect_evaluation(cfg: DictConfig) -> None:
    
    """
    Load, merge, and save the performance_of_ind_wgan in csv format
    Based on the individual scores (G/D) get the top m models and get the detection scores for ensemble model
    """

    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)

    version = cfg.version

    # Running for different window size
    for window in cfg.windows:
        print("Window size:", window)
        cfg.window = window
        model_param = construct_model_cfg(cfg)
        wgan_eval_df = pd.DataFrame()
        # Run load and train for every model
        for model_cfg in model_param:
            model_type = model_cfg.model_type
            # model_id = '_'.join(str(val) for val in model_cfg.values())
            model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
            model_id = f"{model_cfg.model_type}_{window}_{model_id}"
            print("model_id :", model_id)

            if model_type == 'autoencoder': # or model_id != "wgan_4_16_25_2500":
                continue
        
            gen_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"gen_{model_id}_dict.json"
            dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"dis_{model_id}" 

            if Path(f"{gen_file_name}_dict.json").exists() and Path(f"{dis_file_name}_dict.json").exists():
                with open(f"{gen_file_name}_dict.json", "r") as fp: gen_score = json.load(fp)
                with open(f"{dis_file_name}_dict.json", "r") as fp: dis_score = json.load(fp)            
                dis_score = pd.DataFrame(dis_score).T
                gen_score['AUROC'] = dis_score['AUROC'].values.mean()
                gen_score['AUPRC'] = dis_score['AUPRC'].values.mean()
                print(gen_score)
                wgan_eval_df[model_id] = pd.Series(gen_score)
            else:
                print("Following paths do not exist:")
                print(Path(f"{gen_file_name}_dict.json"))
                print(Path(f"{dis_file_name}_dict.json"))


    # Analyze the performance of WGAN
    # Analyze correlation of gen score and disc scores
    wgan_eval_df = wgan_eval_df.astype(float).T
    """
    Add lines to negate the score of Ds/Gs
    # wgan_eval_df['FID'] = - wgan_eval_df['FID']
    # wgan_eval_df['KID'] = - wgan_eval_df['KID']
    """

    wgan_eval_df_scld = pd.DataFrame(StandardScaler().fit_transform(wgan_eval_df), columns=wgan_eval_df.columns)
    wgan_eval_df_scld.index = wgan_eval_df.index
    print(wgan_eval_df_scld)

    wgan_eval_df_scld['GScore'] = wgan_eval_df_scld[["SS_TSTR_Score"]].mean(axis=1) #TODO: Set GScore to --> Single/Multiple Metric
    wgan_eval_df_scld['DScore'] = wgan_eval_df_scld[["AUROC"]].mean(axis=1) #TODO: Set DScore to --> Single/Multiple Metric

    # TODO: Save data and visualize
    ind_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"performance_of_ind_wgan_{cfg.window}.csv" 
    wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by=['AUROC'], ascending=False)
    ind_file_name.parent.mkdir(exist_ok=True, parents=True)
    wgan_eval_df_scld.to_csv(ind_file_name)
    print("wgan_eval_df: \n", wgan_eval_df_scld)



    #----------------------------------------------------------------------
    # Get the upper bound and ensemble score wrt discriminator metric
    #----------------------------------------------------------------------
    pred_dict = {}
    upper_bound = {}
    eval_dict = {}
    
    wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by = ['AUROC'], ascending=False) #TODO: Criteria Metric? DScore?
    for model_count in range(20):
        pred_dict[model_count] = {}

        model_id = wgan_eval_df_scld.index[model_count]
        dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"dis_{model_id}"
        with open(f"{dis_file_name}_dict.json", "r") as fp: dis_score = json.load(fp)

        for attack, data_dict in dis_score.items():
            # print("attack: ", attack)
            auroc = data_dict["AUROC"]
            auprc = data_dict["AUPRC"]
            mean_score = (auroc+auprc)/2

            try:
                # if upper_bound[attack]['mAUC'] < mean_score: #TODO
                if upper_bound[attack]['AUROC'] < auroc:
                    upper_bound[attack]['AUROC'] = auroc
                    upper_bound[attack]['AUPRC'] = auprc
                    upper_bound[attack]['mAUC'] = mean_score
            except:
                upper_bound[attack] = {}
                upper_bound[attack]['AUROC'] = auroc            
                upper_bound[attack]['AUPRC'] = auprc
                upper_bound[attack]['mAUC'] = mean_score

            pred_score = pd.read_csv(f"{dis_file_name}_{attack}_pred.csv", index_col=0)
            pred_score["prediction"] = StandardScaler().fit_transform(pred_score["prediction"].values.reshape(-1, 1)).flatten()

            try:
                pred_dict[model_count][attack] = pred_dict[model_count-1][attack] + pred_score
            except:
                pred_dict[model_count][attack] = pred_score
            
            y_pred_attack = pred_dict[model_count][attack]["prediction"]/(model_count+1)
            y_test_attack = (pred_dict[model_count][attack]["ground_truth"]/(model_count+1)).astype(int)
            auroc_score = roc_auc_score(y_test_attack, y_pred_attack)
            auprc_score = average_precision_score(y_test_attack, y_pred_attack)

            eval_dict_attack =  {
                "Ensemble": model_count + 1,
                "Attack": attack,        
                "AUROC": auroc_score, 
                "AUPRC": auprc_score
                }
            eval_dict[f"{model_count}_{attack}"] = eval_dict_attack
    

    # TODO: Save data and visualize
    ensemmble_df = pd.DataFrame(eval_dict).T
    ens_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ens_detector_ads_{cfg.window}.csv"
    ensemmble_df.to_csv(ens_file_name)
    print("ensemmble_df :", ensemmble_df)

    max_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"max_detector_{cfg.window}.csv"
    upper_bound_df = pd.DataFrame(upper_bound).T
    upper_bound_df["Attack"] = upper_bound_df.index
    upper_bound_df.to_csv(max_file_name)
    print("upper_bound_df :", upper_bound_df)

    #----------------------------------------------------------------------
    # Get the upper bound and ensemble score wrt generator metric
    #----------------------------------------------------------------------
    wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by = ['GScore'], ascending=True)
    pred_dict = {}
    eval_dict = {}

    for model_count in range(20):
        pred_dict[model_count] = {}
        model_id = wgan_eval_df_scld.index[model_count]
        dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"dis_{model_id}"
        with open(f"{dis_file_name}_dict.json", "r") as fp: dis_score = json.load(fp)

        for attack, data_dict in dis_score.items():
            try:
                pred_dict[model_count][attack] = pred_dict[model_count-1][attack] + pd.read_csv(f"{dis_file_name}_{attack}_pred.csv", index_col=0)
            except:
                pred_dict[model_count][attack] = pd.read_csv(f"{dis_file_name}_{attack}_pred.csv", index_col=0)
            
            y_pred_attack = pred_dict[model_count][attack]["prediction"]/(model_count+1)
            y_test_attack = (pred_dict[model_count][attack]["ground_truth"]/(model_count+1)).astype(int)
            try:
                auroc_score = roc_auc_score(y_test_attack, y_pred_attack)
                auprc_score = average_precision_score(y_test_attack, y_pred_attack)
            except Exception as e:
                print(f"Issue with {attack}")
                print(e)

            eval_dict_attack =  {
                "Ensemble": model_count + 1,
                "Attack": attack,        
                "AUROC": auroc_score, 
                "AUPRC": auprc_score
                }
            if auroc_score == 0:
                print("AUROC == 0")
                print("y_test_attack ", y_test_attack)
                print("y_pred_attack : ", y_pred_attack)
                

            eval_dict[f"{model_count}_{attack}"] = eval_dict_attack
 
    # TODO: Save data and visualize
    ensemmble_df = pd.DataFrame(eval_dict).T
    ens_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ens_detector_ags_{cfg.window}.csv"
    ensemmble_df.to_csv(ens_file_name)
    print(ensemmble_df)

# Main function
if __name__ == '__main__':
    run_ens_fixed_detect_evaluation()