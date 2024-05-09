from models.train_models.train_autoencoder import *
from models.train_models.train_wgan import *

def train_ind_model(cfg, model_cfg, model, cbk, remaining_epoch, x_data):
    if remaining_epoch  == 0:
        return None
    if model_cfg.model_type == 'autoencoder':
        history = train_autoencoder_model(cfg, model_cfg, model, cbk, remaining_epoch, x_data)
    elif model_cfg.model_type == 'wgan':
        history = train_wgan_model(cfg, model_cfg, model, cbk, remaining_epoch, x_data)
    return history