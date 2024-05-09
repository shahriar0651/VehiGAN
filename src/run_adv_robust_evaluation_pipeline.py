import os
import hydra
from pathlib import Path
from omegaconf import DictConfig
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import joblib

from dataset import *
from models import *
from helper.helper_fnc import *

@hydra.main(config_path="../config", config_name="config.yaml")
def generate_results(cfg: DictConfig) -> None:
    # Define directory

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
    model_cfg_dict = {}

    # Running for different window size
    print("Window size:", window)
    cfg.window = window
    model_param = construct_model_cfg(cfg)
    dataset_dict = load_data_create_images(cfg) if not cfg.results_only else {}

    # Run load and train for every model
    for model_cfg in model_param:
        model_type = model_cfg.model_type
        model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
        model_id = f"{model_cfg.model_type}_{window}_{model_id}"
        model_cfg_dict[model_id] = model_cfg

    ind_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ind_detector_auroc_{cfg.window}.csv"
    wgan_eval_df_scld = pd.read_csv(ind_file_name, index_col=0)
    print(f"wgan_eval_df_scld: {wgan_eval_df_scld.head()}")

    perf_comb_all = pd.DataFrame([])
    m_max = cfg.m_max

    # Get representative dataset for adversarial attacks
    X_train, y_train, X_test, y_test = get_representative_samples(cfg, dataset_dict, random=cfg.advRandom)
    print("X_train.shape : ", X_train.shape, "X_test.shape : ", X_test.shape)
    X_adv = X_test.copy()
    X_noi = X_test.copy()
    target_indices, colors, opt_factor = get_random_index(cfg, y_test, random=cfg.advRandom)

    model_dict = {}

    for model_count in range(m_max):
        model_id = wgan_eval_df_scld.index[model_count]
        model_cfg = model_cfg_dict[model_id]
        wgan, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
        if remaining_epoch > 0:
            print("Model is not trained yet!")
            continue
        model = wgan.discriminator
        model.compile(optimizer='adam',
            loss= 'binary_crossentropy', #FIXME: BCE or MSE?
            metrics=['accuracy'])
        model_dict[model_id] = model
        print(f"Model Loaded : {model_id}")

        # Attacking individual models----------
        if cfg.advCap in 'indv' or model_count == 0:
            advCap_model = 'White-box'
            print(f"Generating Indv Attacks for Model {model_count}...")
            for indx in tqdm(target_indices[0:]):
                target_img = tf.convert_to_tensor(X_test[indx].reshape((1,window,12,1)))
                if cfg.advFnc == 'fgsm':
                    X_adv[indx] = fgsm_attack(cfg, model, target_img, opt_factor, cfg.epsilon)
                elif cfg.advFnc == 'pgd':
                    X_adv[indx] = pgd_attack(cfg, model, target_img, opt_factor, cfg.epsilon, num_steps=25, step_size=0.001, save = False)
                else:
                    print("Unimplemented Attack!")
                    break
                X_noi[indx] = get_noisy_image(target_img, X_adv[indx])
            save_adversarial_data(cfg, advCap_model, model_id, X_adv, y_test) #TODO: Add model_count to the file name
            print("Adversarial data saved!")

        # Transferring to individual models----------
        elif cfg.advCap == 'trans' and model_count > 0:                
            advCap_model = 'Black-box'

        # Attacking multiple models----------
        elif cfg.advCap == 'multi':
            advCap_model = 'White-box'
            print(f"Generating Multi Attacks for Model {model_count}...")
            for indx in tqdm(target_indices[0:]):
                target_img = tf.convert_to_tensor(X_test[indx].reshape((1,window,12,1)))
                if cfg.advFnc == 'fgsm':
                    X_adv[indx] = fgsm_attack_multi(model_dict, target_img, opt_factor, cfg.epsilon)
                elif cfg.advFnc == 'pgd':
                    X_adv[indx] = pgd_attack_multi(model_dict, target_img, opt_factor, cfg.epsilon)
                else:
                    print("Unimplemented Attack!")
                    break
                X_noi[indx] = get_noisy_image(target_img, X_adv[indx]) #TODO: Add model_count to the file name
            save_adversarial_data(cfg, advCap_model, model_id, X_adv, y_test)
            print("Adversarial data saved!")

        print(f"-------->>>> Model ID: {model_id} <<<<--------")
        print(f"-------->>>> Model Access: {advCap_model} <<<<--------")
        print(f"-------->>>> Getting theshold <<<<--------")
        thresholds = get_threshold(cfg, model_id, model, X_train)
        print(thresholds)
        p_ths = thresholds[f"{cfg.th_percent}"]
        perf_tst = get_performance(model_id, model, X_test, y_test, p_ths)
        perf_tst['advFunc'] = "No Attack"
        perf_noi = get_performance(model_id, model, X_noi, y_test, p_ths)
        perf_noi['advFunc'] = "Random Noise"
        perf_adv = get_performance(model_id, model, X_adv, y_test, p_ths)
        perf_adv['advFunc'] = cfg.advFnc

        # Create a DataFrame from multiple dictionaries
        perf_comb = pd.DataFrame([perf_tst, perf_noi, perf_adv])
        perf_comb['Model'] = model_id
        perf_comb['Epsilon'] = cfg.epsilon
        perf_comb['advType'] = cfg.advType
        perf_comb['advCap'] = cfg.advCap
        perf_comb['modAcc'] = advCap_model
        perf_comb['modCount'] = model_count
        
        perf_comb_all = pd.concat([perf_comb_all,perf_comb], axis = 0, ignore_index=True)
        print("perf_comb_all.shape : ", perf_comb_all.shape)
        
    # Directories...
    ext = f"{cfg.advType}_{cfg.advFnc}_{cfg.advCap}_{cfg.epsilon}"
    data_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / 'advAttacks' / f'adv_performance_drop_{cfg.window}_{ext}.csv'
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    perf_comb_all.to_csv(data_dir, header = True)   
# Main function
if __name__ == '__main__':
    generate_results()

"""
python run_adv_robust_evaluation_pipeline.py version=december_dummy dataset=testing dataset.run_type=unit windows=[10] fast_load=True
python run_adv_robust_evaluation_pipeline.py version=october_24_wisec_8 dataset=testing dataset.run_type=unit windows=[8] fast_load=True results_only=True
python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=unit windows=[10] fast_load=True advCap=indv,trans advType=fp,fn epsilon=0.00,0.005,0.010,0.015,0.020
python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=unit windows=[10] fast_load=True advCap=trans epsilon=0.00,0.010,0.020,0.03,0.04,0.05
python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=unit windows=[10] fast_load=True advCap=trans epsilon=0.00,0.010,0.020,0.03,0.04,0.05
python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False advCap=multi evalType=adversarial epsilon=0.00,0.010
python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=unit windows=[10] fast_load=True advCap=multi evalType=adversarial epsilon=0.010
python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False advCap=multi evalType=adversarial epsilon=0.00,0.005,0.010,0.015,0.020
python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False advCap=indv,trans,multi evalType=adversarial epsilon=0.00,0.005,0.010,0.015,0.020
nohup python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False advCap=indv,trans advFnc='pgd' evalType=adversarial epsilon=0.00,0.005,0.010,0.015,0.020 >/dev/null 2>&1 &
nohup python run_adv_robust_evaluation_pipeline.py -m version=january_icdcs dataset=testing dataset.run_type=full windows=[10] fast_load=False advCap=multi advFnc='pgd' evalType=adversarial epsilon=0.010 >/dev/null 2>&1 &
python run_adv_robust_evaluation_pipeline.py version=january_icdcs dataset=testing dataset.run_type=unit windows=[10] fast_load=True advCap=indv evalType=adversarial m_max=5 advRandom=True epsilon=0.010 advFnc='pgd'
"""
   