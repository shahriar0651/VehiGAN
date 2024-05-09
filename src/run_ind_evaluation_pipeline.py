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
def wgan_evaluate_pipeline(cfg: DictConfig) -> None:
    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir

    print("source_dir ", source_dir)
    print("cfg.dataset.raw_data_dir: ", cfg.dataset.raw_data_dir)

    cfg.dataset.raw_data_dir = source_dir / cfg.dataset.raw_data_dir
    cfg.dataset.clean_data_dir = source_dir / cfg.dataset.clean_data_dir
    
    print("cfg.dataset.raw_data_dir: ", cfg.dataset.raw_data_dir)

    version = cfg.version
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)
    found_wgan = False
    found_ae = False

    # Run model evaluation for WGAN/Autoencoder models
    if cfg.models.model_type != 'baselines':
        
        # Running for different window size
        for window in cfg.windows:
            # print("Window size:", window)
            cfg.window = window

            dataset_dict = load_data_create_images(cfg) if not cfg.results_only else {}

            model_param = construct_model_cfg(cfg)
            wgan_eval_df = pd.DataFrame()
            ae_eval_df = pd.DataFrame()

            # Run load and train for every model
            for model_cfg in model_param:
                model_type = model_cfg.model_type

                model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
                model_id = f"{model_cfg.model_type}_{window}_{model_id}"
                print("model_id :", model_id)

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

                elif model_type == 'wgan':
                    found_wgan = True
                    gen_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"gen_{model_id}"
                    dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{version}' / f"dis_{model_id}"

                    flag_eval_gen = False
                    flag_eval_dis = False

                    if Path(f"{gen_file_name}_dict.json").exists():
                        with open(f"{gen_file_name}_dict.json") as fp: 
                            gen_score = json.load(fp)
                        print("Generator metric loaded")

                    else:
                        flag_eval_gen = True

                    if Path(f"{dis_file_name}_dict.json").exists():
                        with open(f"{dis_file_name}_dict.json") as fp: 
                            dis_score = json.load(fp)
                        print("Discriminator metric loaded!!!")
                    else:
                        flag_eval_dis = True

                    if (flag_eval_dis or flag_eval_gen) and not cfg.results_only:
                        print("Getting model..")
                        wgan, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
                        if remaining_epoch > 0:
                            print("Model not found")
                            continue
                        if flag_eval_dis:
                            dis_score = evaluate_discriminator(cfg, model_cfg, wgan, dataset_dict, dis_file_name)
                            flag_eval_dis = False
                        if flag_eval_gen:
                            gen_score = evaluate_generator(cfg, model_cfg, wgan, dataset_dict, gen_file_name)
                            flag_eval_gen = False

                    
                    if not (flag_eval_gen and flag_eval_dis):
                        dis_score = pd.DataFrame(dis_score).T
                        gen_score['AUROC'] = dis_score['AUROC'].mean()
                        gen_score['AUPRC'] = dis_score['AUPRC'].mean()

                        wgan_eval_df[model_id] = pd.Series(gen_score)
        if found_wgan:
            # Analyze the performance of WGAN
            # Analyze correlation of gen score and disc scores
            wgan_eval_df = wgan_eval_df.astype(float).T
            perf_gen_dis = wgan_eval_df #wgan_eval_df
            perf_gen_dis['DScore'] = np.mean(perf_gen_dis[['AUROC', 'AUPRC']].values, axis = 1)

            perf_gen_dis_scld = pd.DataFrame(StandardScaler().fit_transform(perf_gen_dis), columns=perf_gen_dis.columns)
            perf_gen_dis_scld.index = perf_gen_dis.index
            
            perf_gen_dis_scld['GScore'] = np.mean(perf_gen_dis_scld[["W_Distance", "FID_Score", "KID_Score"]].values, axis = 1)
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
            print(corr_gen_dis)

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
    wgan_evaluate_pipeline()

"""
nohup python run_ind_evaluation_pipeline.py -m version=january_icdcs windows=[8],[10],[12] results_only=false dataset=testing >/dev/null 2>&1 &
nohup python run_ind_evaluation_pipeline.py -m version=january_icdcs windows=[10] results_only=false dataset=testing >/dev/null 2>&1 &
python run_ind_evaluation_pipeline.py -m version=january_icdcs_na windows=[10] results_only=false dataset=testing selected_attacks=["No Attack"]
nohup python run_ind_evaluation_pipeline.py version=january_icdcs models=autoencoder windows=[10] results_only=false dataset=testing >/dev/null 2>&1 &
"""