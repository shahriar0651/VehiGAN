import keras
from pathlib import Path

def train_autoencoder_model(cfg, model_cfg, autoencoder, cbk, remaining_epoch, x_data):
    
    model_root_dir = cfg.models_dir
    model_type = model_cfg.model_type
    window = cfg.window
    num_signals = len(cfg.features)
    num_hid_layers = model_cfg.num_hid_layers
    final_epochs = model_cfg.max_epoch
    model_root_dir = cfg.models_dir

    model_dir = Path(f'{model_root_dir}/{model_type}/{model_type}_{window}_{num_signals}_{num_hid_layers}_{final_epochs}')

    history = autoencoder.fit(x_data, x_data,
                    epochs=remaining_epoch,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    validation_data=(x_data, x_data), callbacks=[cbk])
    model_dir.parent.mkdir(exist_ok=True, parents=True)
    autoencoder.save(model_dir)
    return history