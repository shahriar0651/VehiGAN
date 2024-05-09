import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.get_models.get_autoencoder_model import *
from models.get_models.get_gan_model import *


def load_existing_wgan(cfg, model_cfg):
    EPOCHS = model_cfg.max_epoch
    BATCH_SIZE = cfg.batch_size
    model_root_dir = cfg.models_dir
    model_type = model_cfg.model_type
    window = cfg.window
    num_signals = len(cfg.features)
    num_hid_layers = model_cfg.num_hid_layers
    final_epochs = EPOCHS
    noise_dim = model_cfg.noise_dim


    models_dict = {}

    model_search_pattern = f'{model_root_dir}/{model_type}/generator/{model_type}-generator_{window}_{num_signals}_{num_hid_layers}_{noise_dim}_*'
    model_list = glob.glob(model_search_pattern)
    
    best_epoch = 0
    for model_dir in model_list: 
        epoch_num = int(Path(model_dir).name.split("_")[-1])
        if epoch_num > best_epoch and epoch_num <= final_epochs: 
            best_epoch = epoch_num
    remaining_epoch = final_epochs - best_epoch

    gen_dir = f'{model_root_dir}/{model_type}/generator/{model_type}-generator_{window}_{num_signals}_{num_hid_layers}_{noise_dim}_{best_epoch}'
    disc_dir = f'{model_root_dir}/{model_type}/discriminator/{model_type}-discriminator_{window}_{num_signals}_{num_hid_layers}_{noise_dim}_{best_epoch}'

    models_dict["generator"] = gen_dir
    models_dict["discriminator"] = disc_dir
    print(f"Loaded trained model")

        
    return models_dict, remaining_epoch

def load_existing_ae(cfg, model_cfg):
    EPOCHS = model_cfg.max_epoch
    BATCH_SIZE = cfg.batch_size
    model_root_dir = cfg.models_dir
    model_type = model_cfg.model_type
    window = cfg.window
    num_signals = len(cfg.features)
    num_hid_layers = model_cfg.num_hid_layers
    final_epochs = EPOCHS

    # autoencoder = None
    models_dict = {}

    model_search_pattern = f'{model_root_dir}/{model_type}/{model_type}_{window}_{num_signals}_{num_hid_layers}_*'
    model_list = glob.glob(model_search_pattern)
    
    best_epoch = 0
    for model_dir in model_list: 
        epoch_num = int(Path(model_dir).name.split("_")[-1])
        if epoch_num > best_epoch and epoch_num <= final_epochs: 
            best_epoch = epoch_num
    remaining_epoch = final_epochs - best_epoch

    try:
        model_dir = f'{model_root_dir}/{model_type}/{model_type}_{window}_{num_signals}_{num_hid_layers}_{best_epoch}'
        models_dict["autoencoder"] = model_dir
        print(f"Loaded trained model")
    except:
        print("Creating new model")

    return models_dict, remaining_epoch

def get_ind_model(cfg, model_cfg):

    model_type = model_cfg.model_type
    remaining_epoch = model_cfg.max_epoch

    cbk = None

    if model_type == 'wgan':
        models_dict, remaining_epoch = load_existing_wgan(cfg, model_cfg)
        wgans, cbk = get_wgan(cfg, model_cfg, models_dict)
        return wgans, cbk, remaining_epoch

    elif model_type == 'autoencoder':
        models_dict, remaining_epoch = load_existing_ae(cfg, model_cfg)
        autoencoder, cbk = get_autoencoder(cfg, model_cfg, models_dict)
        return autoencoder, cbk, remaining_epoch