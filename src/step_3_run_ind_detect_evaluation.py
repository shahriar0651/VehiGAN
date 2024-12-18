import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dataset import *
from models import *
from helper import *


@hydra.main(config_path="../config", config_name="config.yaml")
def run_ind_detect_evaluation(cfg: DictConfig) -> None:

    """
    Evaluate individual models and Calculate the correlation
    """
    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    cfg.dataset.raw_data_dir = source_dir / cfg.dataset.raw_data_dir
    cfg.dataset.clean_data_dir = source_dir / cfg.dataset.clean_data_dir
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)

    version = cfg.version
    found_wgan = False
    found_ae = False

    # Run model evaluation for WGAN/Autoencoder models
    if cfg.models.model_type in ['autoencoder', 'wgan']: 
        
        # Running for different window size
        for window in cfg.windows:
            cfg.window = window
            dataset_dict = load_data_create_images(cfg) if not cfg.results_only else {}
            print("dataset_dict: ", dataset_dict.keys())
            model_param = construct_model_cfg(cfg)
            
            wgan_eval_df = pd.DataFrame()
            ae_eval_df = pd.DataFrame()

            # Load and Test Individual Models
            for model_cfg in model_param:
                model_type = model_cfg.model_type

                model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
                model_id = f"{model_cfg.model_type}_{window}_{model_id}"
                print("model_id :", model_id)

                # Evaluate Autoencoder
                if model_type == 'autoencoder':
                    found_ae = True
                    ae_update_flag = False
                    ae_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"ae_{model_id}.json"
                    print("Load or generate eval data for AE")

                    if ae_file_name.exists():
                        with open(ae_file_name) as fp: ae_score = json.load(fp)
                        print("Autoencoder metrics loaded!")
                        ae_update_flag = True
                    elif cfg.results_only == False:
                        autoencoder, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
                        if remaining_epoch == 0:
                            ae_score = evaluate_autoencoder(cfg, model_cfg, autoencoder, dataset_dict, ae_file_name) #TODO: Add later
                            ae_update_flag = True

                    if ae_update_flag:
                        ae_eval_df[model_id] = pd.Series(ae_score)
                
                # Evaluate WGAN
                elif model_type == 'wgan':
                    results_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}'
                    gen_file_name = results_dir / f"gen_{model_id}_dict.json_dict.json" #FIXME: Remove ..._dict.json
                    dis_file_name = results_dir / f"dis_{model_id}_dict.json"
                    
                    found_wgan = True

                    # Load scores or set evaluation flags
                    flag_eval_gen = not gen_file_name.exists()
                    flag_eval_dis = not dis_file_name.exists()

                    if not flag_eval_gen:
                        with open(gen_file_name) as fp:
                            gen_score = json.load(fp)
                        print("Generator metric loaded")
                        flag_eval_gen = len(gen_score) == 0
                    if not flag_eval_dis:
                        with open(dis_file_name) as fp:
                            dis_score = json.load(fp)
                        print("Discriminator metric loaded!!!", dis_score)
                        flag_eval_dis = len(dis_score) == 0  # Re-evaluate if dis_score is empty

                    if (flag_eval_dis or flag_eval_gen) and not cfg.results_only or cfg.reeval_gen or cfg.reeval_dis:
                        print("Getting model..")
                        wgan, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
                        if remaining_epoch > 0:
                            print("Model not found")
                            continue
                        
                        if flag_eval_dis or cfg.reeval_dis:
                            dis_score = evaluate_discriminator(cfg, model_cfg, wgan, dataset_dict, dis_file_name)
                            print("Complete: dis_score", dis_score)

                        if flag_eval_gen or cfg.reeval_gen:
                            gen_score = evaluate_generator(cfg, model_cfg, wgan, dataset_dict, gen_file_name)
                            print("Complete: gen_score", gen_score)

                    else:
                        print("Skipping model evaluation")

                    # Update evaluation DataFrame
                    if gen_score and dis_score:
                        dis_score_df = pd.DataFrame(dis_score).T
                        gen_score['AUROC'] = dis_score_df['AUROC'].mean()
                        gen_score['AUPRC'] = dis_score_df['AUPRC'].mean()

                        # TODO: Update sign of scores
                        gen_score['SS_FAT_Score'] = - gen_score['SS_FAT_Score']
                        gen_score['SS_FAS_Score'] = - gen_score['SS_FAS_Score']
                        gen_score['SS_FAC_Score'] = - gen_score['SS_FAC_Score']
                        # gen_score['TS_AD_Score'] = gen_score['TS_AD_Score']
                        
                        wgan_eval_df[model_id] = pd.Series(gen_score)

        
        if found_wgan:
            # Analyze the performance of WGAN
            # Analyze correlation of gen score and disc scores
            wgan_eval_df = wgan_eval_df.astype(float).T
            perf_gen_dis = wgan_eval_df #wgan_eval_df
            perf_gen_dis['DScore'] = np.mean(perf_gen_dis[['AUROC', 'AUPRC']].values, axis = 1)

            perf_gen_dis_scld = pd.DataFrame(StandardScaler().fit_transform(perf_gen_dis), columns=perf_gen_dis.columns)
            perf_gen_dis_scld.index = perf_gen_dis.index
            
            perf_gen_dis_scld['GScore'] = np.mean(perf_gen_dis_scld[["SS_TSTR_Score"]].values, axis = 1)
            # perf_gen_dis_scld['GScore'] = np.mean(perf_gen_dis_scld[["IM_W_Distance", "IM_FID_Score", "IM_KID_Score"]].values, axis = 1)
            
            perf_gen_dis_scld = perf_gen_dis_scld.sort_values(['GScore'])
            corr_gen_dis = perf_gen_dis_scld.corr() #TODO: Try different methods

            # Directories...
            dir_perf_gen_dis = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f'performance_gen_dis_{cfg.window}.csv'
            dir_corr_gen_dis =  cfg.workspace_dir / 'artifacts' / f'results_{version}' / f'correlation_gen_dis_{cfg.window}.csv'
            # Save data..
            dir_perf_gen_dis.parent.mkdir(parents=True, exist_ok=True)
            perf_gen_dis.to_csv(dir_perf_gen_dis)
            corr_gen_dis.to_csv(dir_corr_gen_dis)
            # ...
            print(perf_gen_dis)
            print(corr_gen_dis.T)

    # Run model evaluation for baseline models
    elif cfg.models.model_type == 'baselines':
        baseline_model_dict =  load_all_baselines(cfg)
        dataset_dict = load_data_create_images(cfg, load_only = True)  
        if cfg.verbose:
            print(f"Dataset loaded with attacks: {dataset_dict.keys()}")
        # Run load and train for every model
        for model_name, model in baseline_model_dict.items():
            baseline_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f'baseline_{model_name}'
            test_baseline(cfg, model_name, model, dataset_dict, baseline_file_name)


# Main function
if __name__ == '__main__':
    run_ind_detect_evaluation()