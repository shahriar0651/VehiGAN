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


def get_prediction(cfg, 
                   model_id,
                   attack = None, 
                   evalType='benign',
                   model_cfg = None,
                   X_test=None,
                   y_test=None,
                   ):
    if evalType=='benign':
        dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / f"dis_{model_id}"
        pred_data = pd.read_csv(f"{dis_file_name}_{attack}_pred.csv", index_col=0)
        pred_score = pred_data["prediction"].values.reshape(-1, 1)
        y_test_attack = pred_data["ground_truth"].astype(int)
    elif evalType=='adversarial':
        model = load_model(cfg, model_cfg)
        pred_score = - model.predict(X_test).flatten()
        y_test_attack = y_test
    p_ths = get_threshold(cfg, model_id)[f"{cfg.th_percent}"]
    y_pred = (pred_score > p_ths).astype(int)
    return pred_score, y_test_attack, y_pred


def get_robust_ens_performance(cfg, wgan_eval_df_scld, 
                               attack = None, 
                               k_max = None, 
                               m_max = None, 
                               evalType=None,
                               model_cfg_dict = None,
                               X_test=None,
                               y_test=None,
                               p_ths = None):
    
    result_dict = {metric: pd.DataFrame() for metric in cfg.metrics}
    pred_scores_all = pd.DataFrame([])
    pred_class_all = pd.DataFrame([])

    for model_count in tqdm(range(m_max)):
        model_id = wgan_eval_df_scld.index[model_count]
        model_cfg = model_cfg_dict[model_id]
        pred_score, y_test_attack, y_pred = get_prediction(cfg, 
                                                   model_id, 
                                                   attack = attack, 
                                                   evalType=evalType, 
                                                   model_cfg = model_cfg,
                                                   X_test=X_test,
                                                   y_test=y_test,)
        
        # scaler = load_scaler(cfg, model_id, scaler_name=cfg.scaler) #TODO: Add loaded scaler 
        # pred_score = scaler.transform(pred_score.reshape(-1, 1)).flatten()
        scaler = StandardScaler()
        pred_score = scaler.fit_transform(pred_score.reshape(-1, 1)).flatten()
        pred_scores_all[model_id] = pred_score
        pred_class_all[model_id] = y_pred.flatten()

    print("pred_scores_all : ", pred_scores_all)
    print("Prediction Loading Complete!")

    n_samples = pred_scores_all.shape[0]
    
    for m in tqdm(range(1, m_max+1), position=0):
        k_max_temp = m 
        if k_max_temp > k_max:
            k_max_temp = k_max

        rng = np.random.default_rng()
        mask = np.ones((n_samples, m), dtype=int) * np.array(range(m))
        mask = rng.permuted(mask, axis=1)

        score_data = pd.DataFrame(
            pred_scores_all.values[np.arange(n_samples)[:, None], mask],
            # index=pred_class_all.index) #TODO : Trying max voting
            index=pred_scores_all.index) #TODO : Trying average voting
        cumulative_avg_df = score_data.expanding(axis=1).mean()

        class_data = pd.DataFrame(
            pred_class_all.values[np.arange(n_samples)[:, None], mask],
            index=pred_class_all.index)
        cumulative_class_df = class_data.expanding(axis=1).mean()

        for k in tqdm(range(1, k_max_temp+1), position=1, leave=False):
            y_score_attack = cumulative_avg_df[k-1]
            auroc = roc_auc_score(y_test_attack, y_score_attack)
            auprc = average_precision_score(y_test_attack, y_score_attack)
            y_class_attack = (cumulative_class_df[k-1] > 0.5).astype(int)
            pre, rec, fscore, support = precision_recall_fscore_support(y_test_attack, y_class_attack, average='binary')
            tpr, fpr, tnr, fnr = calculate_rates(y_test_attack, y_class_attack)
            result_dict['auroc'].loc[k, m] = auroc
            result_dict['auprc'].loc[k, m] = auprc
            result_dict['fpr'].loc[k, m] = fpr
            result_dict['fnr'].loc[k, m] = fnr
            result_dict['pre'].loc[k, m] = pre
            result_dict['rec'].loc[k, m] = rec
            result_dict['fscore'].loc[k, m] = fscore
            
    return result_dict

def get_robust_ens_performance_multi(cfg, wgan_eval_df_scld, 
                               attack = None, 
                               k_max = None, 
                               m_max = None, 
                               evalType=None,
                               model_cfg_dict = None,
                               X_test=None,
                               y_test=None,
                               p_ths = None):
    
    result_dict = {metric: pd.DataFrame() for metric in cfg.metrics}
    pred_scores_all = pd.DataFrame([])
    pred_class_all = pd.DataFrame([])

    for m in tqdm(range(1, m_max+1), position=0):
        model_id = wgan_eval_df_scld.index[m-1]
        pred_scores_all = pd.DataFrame([])
        pred_class_all = pd.DataFrame([])
        scaler_dict = {}
        X_test, y_test = load_adversarial_data(cfg, model_id, advCap_model = 'White-box')
        for model_count in tqdm(range(m)):
            model_id = wgan_eval_df_scld.index[model_count]
            model_cfg = model_cfg_dict[model_id]
            pred_score, y_test_attack, y_pred = get_prediction(cfg, 
                                                    model_id, 
                                                    attack = attack, 
                                                    evalType=evalType, 
                                                    model_cfg = model_cfg,
                                                    X_test=X_test,
                                                    y_test=y_test,)
            
            # scaler = load_scaler(cfg, model_id, scaler_name=cfg.scaler) #TODO: Add loaded scaler 
            # pred_score = scaler.transform(pred_score.reshape(-1, 1)).flatten()
            scaler = StandardScaler()
            pred_score = scaler.fit_transform(pred_score.reshape(-1, 1)).flatten()

            pred_scores_all[model_id] = pred_score
            pred_class_all[model_id] = y_pred
        print("pred_scores_all : ", pred_scores_all.head(5))
        print("Prediction Loading Complete!")
        n_samples = pred_scores_all.shape[0]

        print("m : ", m)
        k_max_temp = m 
        if k_max_temp > k_max:
            k_max_temp = k_max

        rng = np.random.default_rng()
        mask = np.ones((n_samples, m), dtype=int) * np.array(range(m))
        mask = rng.permuted(mask, axis=1)

        score_data = pd.DataFrame(
            pred_scores_all.values[np.arange(n_samples)[:, None], mask],
            index=pred_scores_all.index)
        cumulative_avg_df = score_data.expanding(axis=1).mean()

        class_data = pd.DataFrame(
            pred_class_all.values[np.arange(n_samples)[:, None], mask],
            index=pred_class_all.index)
        cumulative_class_df = class_data.expanding(axis=1).mean()

        for k in tqdm(range(1, k_max_temp+1), position=1, leave=False):
            y_score_attack = cumulative_avg_df[k-1]
            auroc = roc_auc_score(y_test_attack, y_score_attack)
            auprc = average_precision_score(y_test_attack, y_score_attack)
            y_class_attack = (cumulative_class_df[k-1] > 0.5).astype(int)
            pre, rec, fscore, support = precision_recall_fscore_support(y_test_attack, y_class_attack, average='binary')
            tpr, fpr, tnr, fnr = calculate_rates(y_test_attack, y_class_attack)
            result_dict['auroc'].loc[k, m] = auroc
            result_dict['auprc'].loc[k, m] = auprc
            result_dict['fpr'].loc[k, m] = fpr
            result_dict['fnr'].loc[k, m] = fnr
            result_dict['pre'].loc[k, m] = pre
            result_dict['rec'].loc[k, m] = rec
            result_dict['fscore'].loc[k, m] = fscore
    return result_dict

@hydra.main(config_path="../config", config_name="config.yaml")
def generate_results(cfg: DictConfig) -> None:
    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    version = cfg.version
    cfg.window = cfg.windows[0]
    window = cfg.windows[0]
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)
    model_param = construct_model_cfg(cfg)
    model_cfg_dict = {}
    
    for model_cfg in model_param:
        model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
        model_id = f"{model_cfg.model_type}_{window}_{model_id}"
        model_cfg_dict[model_id] = model_cfg
    
    ind_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ind_detector_auroc_{cfg.window}.csv"
    wgan_eval_df_scld = pd.read_csv(ind_file_name, index_col=0)
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
                                                        p_ths = None)
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
    generate_results()

"""
python run_ens_robust_evaluation_pipeline.py version=january_icdcs dataset=testing dataset.run_type=unit windows=[10]
python run_ens_robust_evaluation_pipeline_adv.py -m dataset.run_type=full fast_load=False version=january_icdcs dataset=testing windows=[10] evalType=adversarial advCap=trans epsilon=0.00,0.01
python run_ens_robust_evaluation_pipeline_adv.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False evalType=adversarial advCap=trans epsilon=0.00,0.010 
python run_ens_robust_evaluation_pipeline_adv.py -m version=january_icdcs dataset=testing dataset.run_type=unit windows=[10] fast_load=True evalType=adversarial advCap=multi epsilon=0.00,0.010 
python run_ens_robust_evaluation_pipeline_adv.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False evalType=adversarial advCap=multi epsilon=0.00,0.010
python run_ens_robust_evaluation_pipeline_adv.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False evalType=benign
python run_ens_robust_evaluation_pipeline_adv.py -m version=january_icdcs dataset=testing dataset.run_type=unit windows=[10] fast_load=True advCap=trans,multi evalType=adversarial epsilon=0.010
python run_ens_robust_evaluation_pipeline_adv.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False advCap=indv,trans,multi evalType=adversarial epsilon=0.00,0.005,0.010,0.015,0.020
python run_ens_robust_evaluation_pipeline_adv.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False advCap=trans evalType=adversarial epsilon=0.00,0.005,0.010,0.015,0.020
"""

