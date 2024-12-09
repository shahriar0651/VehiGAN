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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import joblib
from dataset import *
from models import *
from helper import *


@hydra.main(config_path="../config", config_name="config.yaml")
def run_ens_random_robust_evaluation(cfg: DictConfig) -> None:
    # Define directory

    """
    Evaluate ensemble model (m_max / k_max) on benign and adversarial datasets
    
    Create table and save the followigs:
    mean_file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / 'advAttacks' / f"ens_detector_{metric}_mean_{cfg.window}_{ext}.csv"
    std_file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / 'advAttacks' / f"ens_detector_{metric}_std_{cfg.window}_{ext}.csv"
    """

    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    cfg.dataset.raw_data_dir = source_dir / cfg.dataset.raw_data_dir
    cfg.dataset.clean_data_dir = source_dir / cfg.dataset.clean_data_dir
    print("cfg.dataset.clean_data_dir :", cfg.dataset.clean_data_dir )
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)


    version = cfg.version
    cfg.window = cfg.windows[0]
    window = cfg.windows[0]


    model_param = construct_model_cfg(cfg)

     # Load Dataset dict
    dataset_dict = load_data_create_images(cfg) if not cfg.results_only else {}
    X_train, y_train, X_test, y_test = get_representative_samples(cfg, dataset_dict, random=cfg.advRandom)
    
    model_cfg_dict = {}
    for model_cfg in model_param:
        model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
        model_id = f"{model_cfg.model_type}_{window}_{model_id}"
        model_cfg_dict[model_id] = model_cfg
    
    ind_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"performance_of_ind_wgan_{cfg.window}.csv"
    wgan_eval_df_scld = pd.read_csv(ind_file_name, index_col=0)
    wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by=['AUROC'], ascending=False)
    print("wgan_eval_df: \n", wgan_eval_df_scld)
    print("Starting evaluation")
    
    results_total_dict = {metric: [] for metric in cfg.metrics}
    auprc_dfs = []
    fpr_dfs = []

    k_max = cfg.k_max
    m_max = cfg.m_max

    if cfg.evalType == 'benign':
        print("Starting Beningn Evaluation")
        for attack in cfg.selected_attacks[1:]:
            print(f"---------->>> Attack : {attack} <<<--------")
            results_dict = get_robust_ens_performance(cfg, 
                                                      wgan_eval_df_scld, 
                                                      attack = attack, 
                                                      k_max = k_max, 
                                                      m_max = m_max, 
                                                      evalType='benign',
                                                      model_cfg_dict = model_cfg_dict,
                                                      )
    
            for metric in cfg.metrics:
                results_total_dict[metric].append(results_dict[metric].values)
            

    elif cfg.evalType == 'adversarial' and cfg.advCap in ['indv', 'trans']:
        print("Starting Adversarial Evaluation")
        advCap_model = 'White-box'
        target_model_id = wgan_eval_df_scld.index[0]
        X_adv, y_test = load_adversarial_data(cfg, target_model_id, advCap_model)
        results_dict = get_robust_ens_performance(cfg,        
                                                  wgan_eval_df_scld, 
                                                  attack = "Overall", 
                                                  k_max = k_max, 
                                                  m_max = m_max, 
                                                  evalType = 'adversarial', 
                                                  model_cfg_dict = model_cfg_dict,
                                                  X_train=X_train,
                                                  X_test=X_adv,
                                                  y_test=y_test,
                                                  p_ths = None)
        for metric in cfg.metrics:
            results_total_dict[metric].append(results_dict[metric].values)

    elif cfg.evalType == 'adversarial' and cfg.advCap in ['multi']:
        print("Starting Adversarial Evaluation")
        advCap_model = 'White-box'
        target_model_id = wgan_eval_df_scld.index[0]
        # X_adv, y_test = load_adversarial_data(cfg, target_model_id, advCap_model)
        results_dict = get_robust_ens_performance_multi(cfg,  
                                                        wgan_eval_df_scld, 
                                                        attack = "Overall", 
                                                        k_max = k_max, 
                                                        m_max = m_max, 
                                                        evalType = 'adversarial', 
                                                        model_cfg_dict = model_cfg_dict,
                                                        X_train=None,
#                                                       p_ths = None,
                                                        )
        for metric in cfg.metrics:
            results_total_dict[metric].append(results_dict[metric].values)


    indeces = results_dict[metric].index
    columns = results_dict[metric].columns
    ext = f"{cfg.evalType}_{cfg.advCap}_{cfg.epsilon}"

    for metric, metric_dfs in results_total_dict.items():

        mean_df = pd.DataFrame(np.array(metric_dfs).mean(axis=0), index=indeces, columns=columns)
        std_df = pd.DataFrame(np.array(metric_dfs).std(axis=0), index=indeces, columns=columns)

        if metric == 'auroc':
            print(f"\n{metric.upper()}: Mean and Standard Deviation: ")
            print(mean_df, "\n", std_df)

        # Save mean and std to CSV
        mean_file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / 'advAttacks' / f"ens_detector_{metric}_mean_{cfg.window}_{ext}.csv"
        std_file_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / 'advAttacks' / f"ens_detector_{metric}_std_{cfg.window}_{ext}.csv"
        mean_df.to_csv(mean_file_dir)
        std_df.to_csv(std_file_dir)
        
# Main function
if __name__ == '__main__':
    run_ens_random_robust_evaluation()