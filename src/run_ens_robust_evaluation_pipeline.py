import os
import hydra
from pathlib import Path
from omegaconf import DictConfig
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import *
from models import *
from helper import *

@hydra.main(config_path="../config", config_name="config.yaml")
def generate_results(cfg: DictConfig) -> None:
    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    version = cfg.version
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)

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
        
            gen_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"gen_{model_id}"
            dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"dis_{model_id}"

            if Path(f"{gen_file_name}_dict.json").exists() and Path(f"{dis_file_name}_dict.json").exists():
                with open(f"{gen_file_name}_dict.json", "r") as fp: gen_score = json.load(fp)
                with open(f"{dis_file_name}_dict.json", "r") as fp: dis_score = json.load(fp)            
                dis_score = pd.DataFrame(dis_score).T
                gen_score['AUROC'] = dis_score['AUROC'].values.mean()
                gen_score['AUPRC'] = dis_score['AUPRC'].values.mean()
                # gen_score['KID_2'] = abs(complex(gen_score['KID_2']))
                print(gen_score)
                wgan_eval_df[model_id] = pd.Series(gen_score)
                

    # Analyze the performance of WGAN
    # Analyze correlation of gen score and disc scores
    wgan_eval_df = wgan_eval_df.astype(float).T
    wgan_eval_df_scld = pd.DataFrame(StandardScaler().fit_transform(wgan_eval_df), columns=wgan_eval_df.columns)
    wgan_eval_df_scld.index = wgan_eval_df.index
    # gen_metrics = [col for col in wgan_eval_df_scld.columns if col not in ["AUROC", "AUPRC", "EMD"]]
    # wgan_eval_df_scld['Gen_Score'] = wgan_eval_df_scld[gen_metrics].mean(axis=1)
    wgan_eval_df_scld['GScore'] = wgan_eval_df_scld[["W_Distance", "FID_Score", "KID_Score"]].mean(axis=1)
    wgan_eval_df_scld['DScore'] = wgan_eval_df_scld[["AUROC", "AUPRC"]].mean(axis=1)
    # wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by = ['AUROC'], ascending=False)

    wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by = ['AUROC'], ascending=False) #TODO: Metric
    # wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by = ['Gen_Score'], ascending=True)
    print("wgan_eval_df: \n", wgan_eval_df_scld)

    pred_dict = {}
    upper_bound = {}
    eval_dict = {}
    
    #-----------------------------------------
    k_max_cut = 5
    m_max_cut = 20

    auroc_dfs = []
    auprc_dfs = []

    for attack in cfg.selected_attacks[1:]:
        print(f"---------->>> Attack : {attack} <<<--------")
        pred_scores_all = pd.DataFrame([])
        for model_count in range(m_max_cut):
            pred_dict[model_count] = {}
            # print("Model count: ", model_count)
            model_id = wgan_eval_df_scld.index[model_count]
            dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"dis_{model_id}"
            pred_data = pd.read_csv(f"{dis_file_name}_{attack}_pred.csv", index_col=0)
            pred_score = StandardScaler().fit_transform(pred_data["prediction"].values.reshape(-1, 1)).flatten()
            pred_scores_all[model_id] = pred_score
        
        y_test_attack = pred_data["ground_truth"].astype(int)

        n_samples = pred_scores_all.shape[0]
        auroc_df = pd.DataFrame()
        auprc_df = pd.DataFrame()

        for m_max in [1, 2, 3, 5, 10, 15, 20]:
            k_max = m_max 
            if k_max > k_max_cut:
                k_max = k_max_cut

            rng = np.random.default_rng()
            mask = np.ones((n_samples, m_max), dtype=int) * np.array(range(m_max))
            mask = rng.permuted(mask, axis=1)

            data = pd.DataFrame(
                pred_scores_all.values[np.arange(n_samples)[:, None], mask],
                index=pred_scores_all.index)

            cumulative_avg_df = data.expanding(axis=1).mean()

            for k in range(k_max):
                y_pred_attack = cumulative_avg_df[k]
                auroc_score = roc_auc_score(y_test_attack, y_pred_attack)
                auprc_score = average_precision_score(y_test_attack, y_pred_attack)
                auroc_df.loc[k+1, m_max] = auroc_score
                auprc_df.loc[k+1, m_max] = auprc_score
        
        auroc_dfs.append(auroc_df.values)
        auprc_dfs.append(auprc_df.values)

    indeces = auroc_df.index
    columns = auroc_df.columns

    auroc_mean = pd.DataFrame(np.array(auroc_dfs).mean(axis=0), index=indeces, columns=columns)
    auroc_std = pd.DataFrame(np.array(auroc_dfs).std(axis=0), index=indeces, columns=columns)

    auprc_mean = pd.DataFrame(np.array(auprc_dfs).mean(axis=0), index=indeces, columns=columns)
    auprc_std = pd.DataFrame(np.array(auprc_dfs).std(axis=0), index=indeces, columns=columns)
    print("AUROC: Mean and Standard Deviation: ")
    print(auroc_mean, "\n", auroc_std)

    print("AUPRC: Mean and Standard Deviation: ")
    print(auprc_mean, "\n", auprc_std)

    # TODO: Save data and visualize
    file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ens_detector_auroc_mean_{cfg.window}.csv"
    auroc_mean.to_csv(file_dir)
    file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ens_detector_auroc_std_{cfg.window}.csv"
    auroc_std.to_csv(file_dir)
    file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ens_detector_auprc_mean_{cfg.window}.csv"
    auprc_mean.to_csv(file_dir)
    file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ens_detector_auprc_std_{cfg.window}.csv"
    auprc_std.to_csv(file_dir)


# Main function
if __name__ == '__main__':
    generate_results()

"""
python run_ens_robust_evaluation_pipeline.py version=december_dummy dataset=testing dataset.run_type=unit windows=[10]
"""