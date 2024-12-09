import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler
import joblib


def get_list_of_files(data_type: str, clean_data_dir: str):
    file_paths = glob.glob(f"{clean_data_dir}/*.csv")
    print("all file_paths: ", clean_data_dir, file_paths)
    return file_paths

def load_data(file_paths, selected_features, run_id):
    selected_file_paths = []
    for file_path in file_paths:
        file_name = file_path.split("/")[-1].split(".")[0]
        if run_id in file_name:
            selected_file_paths.append(file_path)
    
    print("selected_file_path: ", selected_file_paths)
    df_time = pd.read_csv(selected_file_paths[0], index_col=0)
    print(df_time.columns)
    df_time = df_time[selected_features]
    return df_time

def scale_dataset(data_type, features, labels, df_time, scaler_dir):
    print(data_type)

    feat_code = ("").join([f[0] for f in features])
    scaler_name = f"scaler_{feat_code}.save" 
    scaler_file_dir = scaler_dir / scaler_name

    if data_type == 'training':
        scaler = MinMaxScaler()
        scaler = scaler.fit(df_time[features].values)
        joblib.dump(scaler, scaler_file_dir) 
        print("Scaler saved!!")

    if data_type == 'testing':
        scaler = joblib.load(scaler_file_dir) 

    df_time_scaled_feat = pd.DataFrame(scaler.transform(df_time[features].values), columns=features)
    df_time_scaled = pd.concat([df_time_scaled_feat, df_time[labels]], axis= 1)
    return df_time_scaled


def create_rolling_windows_stack(array, window_size, step_size=1):
    num_windows = (array.shape[0] - window_size) // step_size + 1
    shape = (num_windows, window_size, array.shape[1])
    strides = (step_size * array.strides[0], array.strides[0], array.strides[1])
    stacked_windows =  np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return stacked_windows


# def create_rolling_windows_stack(array, window_size):
#     num_windows = array.shape[0] - window_size + 1
#     shape = (num_windows, window_size, array.shape[1])
#     strides = (array.strides[0],) + array.strides
#     stacked_windows = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
#     return stacked_windows

def create_image_labels(df_time, window, features, labels, selected_features):

    x_data_all = np.array([])
    y_data_all = np.array([])
    
    for vehicle_id in df_time['id'].unique():
        df_vehicle = df_time[df_time['id'] == vehicle_id][selected_features]
        list_of_chunks = df_vehicle['time_chunk'].unique()

        for chunk in list_of_chunks:

            df_vehicle_chunk = df_vehicle[df_vehicle['time_chunk'] == chunk]
            # print(f"Before: {df_vehicle_chunk.shape}")
            # df_vehicle_chunk = df_vehicle_chunk.dropna()
            # print(f"After: {df_vehicle_chunk.shape}")

            if df_vehicle_chunk.shape[0] < window:
                continue
            x_data = create_rolling_windows_stack(df_vehicle_chunk[features].values, window)
            y_data = create_rolling_windows_stack(df_vehicle_chunk[['attack_gt']].values, window)
            try:
                x_data_all = np.concatenate((x_data_all, x_data), axis = 0)
                y_data_all = np.concatenate((y_data_all, y_data), axis = 0)
            except:
                x_data_all = x_data
                y_data_all = y_data

    x_data_all = x_data_all.reshape(-1, window, len(features), 1)
    y_data_all = np.mean(np.squeeze(y_data_all), axis = 1)
    return x_data_all, y_data_all


def load_data_create_images(cfg, load_only = False):

    dataset_dict = {}

    data_type = cfg.dataset.data_type
    raw_data_dir = cfg.dataset.raw_data_dir
    clean_data_dir = cfg.dataset.clean_data_dir
    print("clean_data_dir :", clean_data_dir)
    run_id = cfg.dataset.run_id
    window = cfg.window
    features = cfg.features
    labels = cfg.labels
    selected_features = features + labels
    run_type = cfg.dataset.run_type
    selected_attacks = cfg.selected_attacks
    scaler_dir = cfg.scaler_dir

    if run_type == 'unit':
        num_of_samples = int(cfg.dataset.unit_samples)

    print("clean_data_dir: ", clean_data_dir)
    
    file_paths = get_list_of_files(data_type, clean_data_dir)
    df_time = load_data(file_paths, selected_features, run_id)
    df_time = scale_dataset(data_type, features, labels, df_time, scaler_dir)
    print(df_time.columns)

    print("Total attacks in the dataset: ", df_time["attack_name"].unique().shape)
    if cfg.fast_load == True:
        selected_attacks = selected_attacks[0:5]
    for attack in tqdm(selected_attacks):
        df_attack = df_time[df_time['attack_name'] == attack].reset_index(drop = True)
        if df_attack.shape[0] == 0:
            continue
        print("Attack Name: ", attack)
        
        if cfg.fast_load == True: #Only for testing
            df_attack = df_attack.iloc[0:int(num_of_samples*2)]

        if load_only == False:
            x_data, y_data = create_image_labels(df_attack, window, features, labels, selected_features)
            index = np.arange(x_data.shape[0])
            if run_type == 'unit':
                num_of_samples = min(num_of_samples, len(index))
                selected_indices = np.random.choice(index, size=num_of_samples, replace=False)
            else:
                selected_indices = index
            dataset_dict[attack] = {"x_data": x_data[selected_indices], "y_data": y_data[selected_indices]}
        else:
            index = np.arange(df_attack.shape[0])
            if run_type == 'unit':
                num_of_samples = min(num_of_samples, len(index))
                selected_indices = np.random.choice(index, size=num_of_samples, replace=False)
            else:
                selected_indices = index
            dataset_dict[attack] = df_attack.iloc[selected_indices]
    return dataset_dict