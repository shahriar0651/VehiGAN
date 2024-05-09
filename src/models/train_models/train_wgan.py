# import keras
import json
from pathlib import Path

def train_wgan_model(cfg, model_cfg, wgan, cbk, remaining_epoch, x_data):  
    
    BATCH_SIZE = cfg.batch_size
    loss_func = cfg.loss.ae
    model_type = model_cfg.model_type
    window = cfg.window
    num_signals = len(cfg.features)
    num_hid_layers = model_cfg.num_hid_layers
    EPOCHS = remaining_epoch
    final_epochs = model_cfg.max_epoch
    model_root_dir = cfg.models_dir
    noise_dim = model_cfg.noise_dim

    generator_dir = f'{model_root_dir}/{model_type}/generator/{model_type}-generator_{window}_{num_signals}_{num_hid_layers}_{noise_dim}_{final_epochs}'
    discriminator_dir = f'{model_root_dir}/{model_type}/discriminator/{model_type}-discriminator_{window}_{num_signals}_{num_hid_layers}_{noise_dim}_{final_epochs}'
    starting_epoch = final_epochs - remaining_epoch
    history_dir = Path(f'{model_root_dir}/{model_type}/history/{model_type}-history_{window}_{num_signals}_{num_hid_layers}_{noise_dim}_{starting_epoch}_{final_epochs}.json')
    history_dir.parent.mkdir(parents=True, exist_ok=True)


    history = wgan.fit(x_data, batch_size=BATCH_SIZE, epochs = EPOCHS, callbacks=[cbk], use_multiprocessing=True) # workers=4
    wgan.generator.save(generator_dir)
    wgan.discriminator.save(discriminator_dir)
    #Save history
    with open(history_dir, "w") as fp:
        json.dump(history.history, fp, indent=4)
    return history.history