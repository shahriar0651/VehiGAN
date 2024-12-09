import os
from pathlib import Path

import hydra
import argparse
import yaml
from omegaconf import DictConfig

from dataset import *
from models import *
from helper import *



@hydra.main(config_path="../config", config_name="config.yaml")
def run_ind_training_pipeline(cfg: DictConfig) -> None:

    #TODO: Set memory grouth....
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # gpus = tf.config.experimental.list_physical_devices
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    #......................

    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir 
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    cfg.dataset.raw_data_dir = source_dir / cfg.dataset.raw_data_dir
    cfg.dataset.clean_data_dir = source_dir / cfg.dataset.clean_data_dir
    if not cfg.models_dir.exists(): os.makedirs(cfg.models_dir)
    if not cfg.scaler_dir.exists(): os.makedirs(cfg.scaler_dir)

    
    # Run model training for WGAN or Autoencoder
    if cfg.models.model_type != 'baselines':
        # Running for different window size
        for window in cfg.windows:
            print("Window size: ", window)
            cfg.window = window
            # Load training data
            dataset_dict = load_data_create_images(cfg)    
            if cfg.verbose:
                print(f"Dataset loaded with attacks: {dataset_dict.keys()}")
            model_param = construct_model_cfg(cfg)

            # Run load and train for every model
            for model_id, model_cfg in enumerate(model_param):
                model_type = model_cfg.model_type
                print(f"\n\nStrating: {model_type}_{model_id}")
                # Load model 
                model, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
                # Train model
                if remaining_epoch > 0:
                    print(f"Training model for remaining {remaining_epoch} epochs")
                    history = train_ind_model(cfg, model_cfg, model, cbk, remaining_epoch, dataset_dict["No Attack"]["x_data"]) #TODO: Fix train_ind_model
                else:
                    print("Model is already trained.")

    # Run model training for baseline models
    elif cfg.models.model_type == 'baselines':
        if cfg.dataset.data_type == 'training':
            dataset_dict = load_data_create_images(cfg, load_only = True)  
            X_train = dataset_dict["No Attack"][cfg["features"]].values
            y_train = dataset_dict["No Attack"]["attack_gt"].values
            # Load training data
            if cfg.verbose:
                print(f"Dataset loaded with attacks: {dataset_dict.keys()}")
                print("Shape of benign dataset: ", X_train.shape, y_train.shape)
            baseline_model_dict =  get_all_baselines(cfg)
            train_all_baselines(cfg, baseline_model_dict, X_train)

# Main function
if __name__ == '__main__':
    run_ind_training_pipeline()

