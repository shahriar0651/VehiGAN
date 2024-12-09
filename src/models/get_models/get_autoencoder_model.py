import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import glob
from pathlib import Path
from tensorflow import keras
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.callbacks import EarlyStopping


def create_autoconder(cfg, model_cfg):
    EPOCHS = model_cfg.max_epoch
    BATCH_SIZE = cfg.batch_size
    model_root_dir = cfg.models_dir
    model_type = model_cfg.model_type
    window = cfg.window
    num_signals = len(cfg.features)
    num_hid_layers = model_cfg.num_hid_layers
    final_epochs = EPOCHS

    n_filters_dict = {
        3 : [128, 64, 32],
        4:  [256, 128, 64, 32],
        5:  [256, 256, 128, 64, 32]}

    n_filters_list = n_filters_dict[num_hid_layers]

    crop_factors = {}
    crop_factors [8] = ((4, 4), (2, 2))
    crop_factors [10] = ((3, 3), (2, 2))
    crop_factors [12] = ((2, 2), (2, 2))



    input_img = keras.Input(shape=(window, num_signals, 1))  
    x = ZeroPadding2D(padding=2)(input_img)
    for n_filters in n_filters_list:    
        print(n_filters)
        x = Conv2D(n_filters, (2, 2), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(input_img, x)


    # encoded_img = keras.Input(shape= x.shape[1][1:])  
    encoded_img = keras.Input(shape= x.shape[1:])  
    x = encoded_img
    # x = keras.layers.Reshape((2, 2, 4))(x)
    for n_filters in n_filters_list[::-1]:    
        x = Conv2D(n_filters, (2, 2), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = Cropping2D(cropping= crop_factors [window])(x)
    decoder = keras.Model(encoded_img, decoded)
    # return encoder, decoder
    # define input to the model:
    x = keras.Input(shape=(window, num_signals, 1))
    autoencoder = keras.Model(x, decoder(encoder(x)))
    return autoencoder 

def get_autoencoder(cfg, model_cfg, models_dict=None):
    
    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        try:
            autoencoder = keras.models.load_model(models_dict["autoencoder"])
        except:
            autoencoder = create_autoconder(cfg, model_cfg)
        
        # compile autoencoder
        opt = keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5, beta_2=0.99)
        autoencoder.compile(loss='mse', optimizer=opt, metrics=['mse'])
        autoencoder.summary()
        # patient early stopping
        cbk = EarlyStopping(monitor='val_mse', mode="auto", verbose=1, patience=25)

    return autoencoder, cbk
