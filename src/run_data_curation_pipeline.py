from pathlib import Path
import hydra
from omegaconf import DictConfig
from dataset import *
from helper import *
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


def get_feature_lists(cfg):
    # ------------------
    # Creating the dictionary...
    vectors = cfg.vectors
    rename_dict = {cfg.time_feat: "time", cfg.label: "label"}
    decom_func_dict = {"x": np.cos, "y": np.sin}
    vehicle_feats_list = [f"{cfg.vehicle_type}_{feat}" for feat in cfg.vehicle_feats]
    vehicle_feats_list += cfg.file_attr
    # print("vehicle_feats_list:\n", vehicle_feats_list)

    # list of vector features...
    vector_feats_list = [f"{cfg.vehicle_type}_{vector}" for vector in vectors]
    # print("vector_feats: ", vector_feats_list)
    # Decomposition dictionary functions...
    decom_feat_dict = {}
    for feat_name in vector_feats_list:
        for axis, func in decom_func_dict.items():
            new_feat_name = "_".join([feat_name, axis])
            decom_feat_dict[new_feat_name] = [feat_name, func]

    # List of features after vector decompostions...
    decom_feats_list = list(decom_feat_dict.keys())
    decom_feats_list += [
        f"{cfg.vehicle_type}_pos_x",
        f"{cfg.vehicle_type}_pos_y",
        f"{cfg.vehicle_type}_pos_z",
    ]
    return rename_dict, decom_feat_dict, vehicle_feats_list, decom_feats_list
    # rename_dict, decom_feats_list = get_feature_lists(cfg)

def fix_attack_values(df_raw, faulty_feat):
    # Sample Series with NaN values
    nan_index = df_raw[faulty_feat].isna()
    print(f"Before: Total missing values: {df_raw[faulty_feat].isna().sum()}")

    # Calculate the minimum and maximum values in the Series, ignoring NaN
    min_value = df_raw[faulty_feat].min(skipna=True)
    max_value = df_raw[faulty_feat].max(skipna=True)
    print(f"Min: {min_value}, Max: {max_value}")

    # Replace NaN values with random values in the range [min_value, max_value]
    # You can use np.random.uniform() to generate random values within the range.
    random_values = np.random.uniform(min_value, max_value, df_raw[faulty_feat].isna().sum())
    df_raw[faulty_feat][nan_index] = random_values
    print(f"After: Total missing values: {df_raw[faulty_feat].isna().sum()}")
    return df_raw

@hydra.main(config_path="../config", config_name="config.yaml")
def create_clean_dataset(cfg: DictConfig) -> None:
    # Define directory
    source_dir = Path(__file__).resolve().parent
    print("source_dir ", source_dir)
    # print("cfg.dataset.raw_data_dir: ", cfg.dataset.raw_data_dir)
    cfg.dataset.raw_data_dir = source_dir / cfg.dataset.raw_data_dir
    cfg.dataset.clean_data_dir = source_dir / cfg.dataset.clean_data_dir
    print("cfg.dataset.raw_data_dir: ", cfg.dataset.raw_data_dir)

    feat_not_for_scaling = cfg.feat_not_for_scaling

    file_dirs = glob.glob(f"{cfg.dataset.raw_data_dir}/*.csv")
    file_names = [x.split("/")[-1].split(".")[0][0:] for x in file_dirs]

    print(f"\nTotal Files:\n {len(file_dirs)}, etc.")

    # %%

    # New lists for files and directions...
    file_names_updated = []
    file_dirs_updated = []

    # Filtering the files according to the filters............................
    for index, (file_name, file_dir) in enumerate(zip(file_names, file_dirs)):
        if cfg.dataset.run_id in file_name:
            file_names_updated.append(file_name)
            file_dirs_updated.append(file_dir)

    print(f"\nTotal selected files:\n {len(file_dirs_updated)}, etc.")

    (
        rename_dict,
        decom_feat_dict,
        vehicle_feats_list,
        decom_feats_list,
    ) = get_feature_lists(cfg)

    # Defining a new dataframe...
    df_time = pd.DataFrame([])
    break_flag = False
    # --------------------------------------------------------------------------------------------
    # Reading files in order...
    for index, (file_name, file_dir) in tqdm(enumerate(zip(file_names_updated, file_dirs_updated))):
        try:
            print(f"Loading {file_name}")
            # File information...
            attack_index = file_name.split("-")[2]

            # Load dataset...
            df_time_file = pd.DataFrame([])
            df_raw = pd.read_csv(file_dir)
    
            #TODO: Need to fix VASP to have the correct names
            attack_name_fix = {'RandomOffset': 'RandomPositionOffset',
                               'ConstantOffset' : 'ConstantPositionOffset'}
            df_raw.replace(attack_name_fix, inplace=True)

            attack_to_fix = {'RandomSpeed' : 'rv_speed', 
                             'RandomAcceleration' : 'rv_accel'}
            for faulty_attack, faulty_feat in attack_to_fix.items():
                if faulty_attack in df_raw['attack_type'].values:
                    print(f"Fixing {faulty_attack} attack")
                    df_raw = fix_attack_values(df_raw, faulty_feat)

            print(f"Value count of attack types: {df_raw.attack_type.value_counts()}")
            print(f"Shape of the raw dataset after loading: {df_raw.shape}")
            df_raw = df_raw.dropna()
            print(f"Shape of the raw dataset after dropna: {df_raw.shape}")

            # df_raw['rv_accel'] = df_raw['rv_accel'].rolling(2).mean().fillna(0)

            # unique rv_id and hv_id values...
            rv_id_unique = df_raw.rv_id.unique()
            hv_id_unique = df_raw.hv_id.unique()

            # Considered list of vehicles...
            list_of_vehicle_ids = rv_id_unique

            # Determine the name of the attack----
            attack_type_list = list(df_raw.attack_type.unique())
            attack_type_list.remove("Genuine")
            # -------------------------------------
            if len(attack_type_list) == 0:
                attack_name = "No Attack"
            elif len(attack_type_list) == 1:
                attack_name = attack_type_list[0]
            else:
                attack_name = f"Multiple_{index}"
            print(f"Attack name: {attack_name}")
            # ------------------------------------
            # Converting string to Labels...
            df_raw["attack_gt"] = (df_raw.attack_type != "Genuine").astype(int)
            attack_gt_count = df_raw["attack_gt"].value_counts()
            print(f"Value count of attack_gt: {attack_gt_count}")
            df_raw["attack_index"] = attack_index
            df_raw["attack_name"] = attack_name
            # ------------------------------------------------------

            if attack_name not in cfg.selected_attacks + ["No Attack"]:
                print(f"\n\n{attack_name} is not on the list! Skipping!!!")
                continue
            else:
                print(f"{attack_name} is on the list! Proceeding.......")

            for vehicle_id in tqdm(list_of_vehicle_ids[:]):
                # for vehicle_id in  tqdm([261]):

                # Creaking a chunk data for target vehicle............................
                # Creating filter for target vehicle and taking the chunk of dataframe...
                filter_per_vehicle = df_raw[f"{cfg.vehicle_type}_id"] == vehicle_id
                df_raw_vehicle = df_raw[vehicle_feats_list].loc[filter_per_vehicle]

                # Rounding the time features and removing the duplicates.............
                df_raw_vehicle[cfg.time_feat] = np.round(df_raw_vehicle[cfg.time_feat], 2)
                df_raw_vehicle = df_raw_vehicle.drop_duplicates(subset=cfg.time_feat, keep="last")
                df_raw_vehicle = df_raw_vehicle.sort_values(by=[cfg.time_feat], ascending=True)
                df_raw_vehicle = df_raw_vehicle.reset_index(drop=True)

                # Looking for discontinuty in time features...
                high_time_jump = df_raw_vehicle[cfg.time_feat].diff().fillna(0.10) > 0.15
                high_time_jump = high_time_jump.reset_index(drop=True)
                indices_of_disc_ = high_time_jump.loc[high_time_jump].index.to_list()

                # Starting and ending index...
                start_index = high_time_jump.index[0]
                end_index = high_time_jump.index[-1] + 1
                # Adding starting and ending index to the list...
                indices_of_disc = [start_index] + indices_of_disc_ + [end_index]
                # print("indices_of_disc: ", indices_of_disc)

                df_raw_vehicle["time_chunk"] = 0
                # Adding time chunk info for each of the datasegment...
                for i in range(len(indices_of_disc) - 1):
                    # Indices of boundaries...
                    from_index, to_index = indices_of_disc[i], indices_of_disc[i + 1]
                    # print("from_index, to_index: ", from_index, to_index)
                    # Defining a new array for time-chunk information...
                    df_raw_vehicle["time_chunk"].iloc[from_index:to_index] = i + 1

                    # Adding vector features...
                    for i, (new_feat, [vector, func]) in enumerate(decom_feat_dict.items()):
                        df_raw_vehicle[new_feat] = (
                            df_raw_vehicle[vector] * (-1) ** (i) * func(df_raw_vehicle[f"{cfg.vehicle_type}_heading"])
                        )

                    # Adding delta features...
                    for delta_feat in decom_feats_list:
                        df_raw_vehicle["del_" + delta_feat] = df_raw_vehicle[delta_feat].diff().bfill()
                    # -----------------------------
                df_raw_vehicle_clean = df_raw_vehicle.drop(indices_of_disc_).copy()

                # Appending to the new dataframe...
                df_time_file = pd.concat([df_time_file, df_raw_vehicle_clean], ignore_index=True)
        except Exception as e:
            print(e)
            print("Skipping file: ", file_name)

        # feature rename dict......................................................
        dict_of_feats_rename = {}

        # Creating the list of delta features...
        for column in df_time_file.columns:
            if cfg.vehicle_type in column:
                dict_of_feats_rename[column] = column.replace(cfg.vehicle_type + "_", "")

        # Rename dataframe.............................................
        df_time_file = df_time_file.rename(columns=dict_of_feats_rename)
        # Merging the dataframe.........................................
        df_time = pd.concat([df_time, df_time_file], ignore_index=True)
        # ----------------------------------------------------------------

        # Plot correlation heatmap........................................
        feat_for_scaling = sorted(list(set(list(df_time.columns)) - set(feat_not_for_scaling)))
        # Converting data type for scalable features...
        df_time_file[feat_for_scaling] = df_time_file[feat_for_scaling].astype(float)
        # --------------------------------------------------------------------

        # Scaling the dataset...
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        df_time_file_scaled = pd.DataFrame(
            scaler.fit_transform(df_time_file[feat_for_scaling].values),
            columns=feat_for_scaling,
        )

        df_time_file_scaled = pd.concat([df_time_file_scaled, df_time_file[feat_not_for_scaling]], axis=1, ignore_index=False)

        # Plot the correlation heatmap..........
        plot_dir = Path(f"{source_dir}/../artifacts/plots/data_analysis//")
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(4, 3))
        sns.heatmap(
            df_time_file_scaled[cfg.vis_feat].corr(),
            linewidths=1,
            alpha=0.85,
            cmap="viridis",
        )
        plt.title(f"Corr coeffs: {attack_name}")
        plt.tight_layout()
        plt.savefig(
            f"{plot_dir}/heatmap_file_{attack_name}_{cfg.dataset.run_id}.jpg",
            dpi=250,
        )
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")
    # ------------------------------------------------------------------

    print("Before: ", df_time.shape[0])
    df_time = df_time.dropna()
    print("After: ", df_time.shape[0])

    # Saving the data----------------------------------------------------------------------------------
    df_time.reset_index(drop=True, inplace=True)
    # --------------------------------------------------------------------------------------------------
    merged_file_name = f"merged_{cfg.dataset.data_type}_{cfg.dataset.run_id}.csv"
    data_directory = Path(f"{cfg.dataset.clean_data_dir}/{merged_file_name}")
    data_directory.parent.mkdir(parents=True, exist_ok=True)
    df_time.to_csv(data_directory, header=True, index=True)
    # ---------------------------------------------------------------------------------------------------


# Main function
if __name__ == "__main__":
    create_clean_dataset()