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
def run_ind_robustness_evaluation(cfg: DictConfig) -> None:
    """
    Generate adversarial samples and evaluate "INDIVIDUAL" model on Benign, Adversarial, and Noisy samples
    Save results with the following:
        "precision": pre,
        "recall": rec,
        "fscore": fscore,
        "support": support,
        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "FNR": tnr
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
    window = cfg.windows[0]
    cfg.window = window

    model_cfg_dict = {}
    perf_comb_all = pd.DataFrame([])
    m_max = cfg.m_max

    # Load performance_of_ind_wgan sorted wrt AUROC in step 4
    ind_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"performance_of_ind_wgan_{cfg.window}.csv"
    wgan_eval_df_scld = pd.read_csv(ind_file_name, index_col=0)
    wgan_eval_df_scld = wgan_eval_df_scld.sort_values(by=['AUROC'], ascending=False)
    print(f"wgan_eval_df_scld: {wgan_eval_df_scld.head()}")

    # Load Dataset dict
    dataset_dict = load_data_create_images(cfg) if not cfg.results_only else {}
    X_train, y_train, X_test, y_test = get_representative_samples(cfg, dataset_dict, random=cfg.advRandom)
    print("Dataset Loaded!")
    # Get representative dataset for adversarial attacks
    target_indices, colors, opt_factor = get_random_index(cfg, y_test, random=cfg.advRandom)
    print("X_train.shape : ", X_train.shape, "X_test.shape : ", X_test.shape)
    X_adv = X_test.copy()
    X_noi = X_test.copy()


    # Load Model Config
    model_param = construct_model_cfg(cfg)
    for model_cfg in model_param:
        model_type = model_cfg.model_type
        model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
        model_id = f"{model_cfg.model_type}_{window}_{model_id}"
        model_cfg_dict[model_id] = model_cfg


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
            # X_adv is already generated with model_count == 0 

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
        p_ths = thresholds[f"{cfg.th_percent}"]
        perf_tst = get_performance(model_id, model, X_test, y_test, p_ths)
        perf_tst['advFunc'] = "NoAttack"
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
        # if perf_tst["fscore"] > 0.00:
        #     perf_tst_df.loc[noise_dim, num_hid_layers] = perf_tst["fscore"]
        #     perf_noi_df.loc[noise_dim, num_hid_layers] = perf_noi["fscore"]
        #     perf_adv_df.loc[noise_dim, num_hid_layers] = perf_adv["fscore"]
        #     perf_drp_df.loc[noise_dim, num_hid_layers] = perf_tst["fscore"] - perf_adv["fscore"]
        
    # Directories...
    ext = f"{cfg.advType}_{cfg.advFnc}_{cfg.advCap}_{cfg.epsilon}"
    data_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / 'advAttacks' / f'adv_performance_drop_{cfg.window}_{ext}.csv'
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    perf_comb_all.to_csv(data_dir, header = True)   
# Main function
if __name__ == '__main__':
    run_ind_robustness_evaluation