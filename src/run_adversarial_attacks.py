import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd
import copy
from tqdm import tqdm
# Import the attack
# from cleverhans.future.tf2.attacks import fast_gradient_method

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from dataset import *
from models import *
from helper import *

# Function to perform FGSM attack on the discriminator
def fgsm_attack(model, image, factor, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss =  - factor * tf.reduce_mean(prediction)   #FIXME For maximization (minimize the negative prediction)
        # loss = tf.keras.metrics.mean_squared_error(mean_ben_score, prediction)   #FIXME For maximization (minimize the negative prediction)
    
    gradient = tape.gradient(loss, image)
    perturbation = epsilon * tf.sign(gradient)
    adv_image = image + perturbation

    # Assuming your initial tensor is 'initial_tensor'
    initial_shape = perturbation.shape
    perturbation_flat = tf.reshape(perturbation, [-1])
    tf.random.set_seed(42)
    noise = tf.random.shuffle(perturbation_flat)
    noise = tf.reshape(noise, initial_shape)
    noisy_image = image + noise

    # adv_image = tf.clip_by_value(adv_image, 0, 1)  # Clip values to [0, 1] range
    return adv_image.numpy(), noisy_image.numpy()

def plot_hist_with_attack(pred_df, original_data, adv_data, random_index, model_id, colors, plot_dir, advattack):
    # Create the histogram plot using seaborn
    plt.figure(figsize=(6, 4))
    ax = sns.histplot(pred_df, hue="Label", x="Prediction")
    boundaries = (pred_df['Prediction'].min(), 
                  pred_df['Prediction'].max(), 
                  original_data["Pred"], 
                  adv_data["Pred"])
    x_min = min(boundaries)
    x_max = max(boundaries)
    
    # Plot an arrow for the chosen point
    ax.annotate('Org', 
                xy=(original_data["Pred"], 0),  # Arrow starting point
                xytext=(original_data["Pred"], 1500),  # Text starting point
                arrowprops=dict(facecolor=colors[0], shrink=0.05),  # Arrow properties
                fontsize=10,
                color='black',
                horizontalalignment='center',  # Text alignment
                verticalalignment='bottom',  # Text alignment
                )

    # Plot an arrow for the chosen point
    ax.annotate('Adv', 
                xy=(adv_data["Pred"], 0),  # Arrow starting point
                xytext=(adv_data["Pred"], 1500),  # Text starting point
                arrowprops=dict(facecolor=colors[1], shrink=0.05),  # Arrow properties
                fontsize=10,
                color='black',
                horizontalalignment='center',  # Text alignment
                verticalalignment='bottom',  # Text alignment
                )
    # Show the plot
    ax.set_xlabel("Anomaly Score")
    ax.set_title(f"{model_id} with attack {random_index}")
    ax.set_xlim(x_min, x_max)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{advattack}_advatt_{model_id}_{random_index}_dist.jpg")
    plt.show()

def show_attack_images(original_data, adv_data, adv_perturb, random_index, model_id, plot_dir, advattack):
    # Create a single figure with three subplots
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    # Original Image
    sns.heatmap(np.reshape(original_data['Img'], (10, 12)), ax = axes[0]) #, vmin = 0, vmax = 1, 
    axes[0].set_title("Original Input: {}".format(original_data['Pred']))
    axes[0].set_xlabel("Original Label: {}".format(original_data['Lab']))

    # Adversarial Example
    sns.heatmap(np.reshape(adv_data['Img'], (10, 12)), ax = axes[1])
    axes[1].set_title("Adversarial Input: {}".format(adv_data["Pred"]))
    axes[1].set_xlabel("Original Label: {}".format(original_data["Lab"]))

    # Adversarial Perturbation
    sns.heatmap(np.reshape(adv_perturb, (10, 12)), ax = axes[2])
    axes[2].set_title("Adversarial Perturbation")
    fig.suptitle(f"{model_id} with attack {random_index}")
    plt.tight_layout()
    # Save the entire figure to a single file
    plt.savefig(f"{plot_dir}/{advattack}_advatt_{model_id}_{random_index}_data.jpg")

def get_random_index(y_test, advattack):
    if advattack == 'fn':
        target_indices = np.where(y_test == 1)[0]   
        colors = ['red', 'green']    
        opt_factor = -1   
    if advattack == 'fp':
        target_indices = np.where(y_test == 0)[0] 
        colors = ['green', 'red'] 
        opt_factor = 1   
    np.random.shuffle(target_indices)
    return target_indices, colors, opt_factor

def get_threshold(model, X_train, percent):
    p_train = - model.predict(X_train).flatten()
    p_ths = np.percentile(p_train, percent)
    return p_ths
def get_performance(model_id, model, X_test, y_test, attack, p_ths):
    prediction = - model.predict(X_test).flatten()
    y_pred = (prediction > p_ths).astype(int)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
    performance = {"precision" : precision, 
                   "recall" : recall,
                   "fscore" : fscore,
                   "support" : support}   
    print(f"{model_id} : ", performance)
    return performance

@hydra.main(config_path="../config", config_name="config.yaml")
def adv_evaluate_pipeline(cfg: DictConfig) -> None:
    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    cfg.dataset.raw_data_dir = source_dir / cfg.dataset.raw_data_dir
    cfg.dataset.clean_data_dir = source_dir / cfg.dataset.clean_data_dir
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)
    result_dir = Path(f'{cfg.workspace_dir}/artifacts/results_{cfg.version}/')
    plot_dir = Path(f"{result_dir}/advAttacks")
    plot_dir.mkdir(parents=True, exist_ok=True)
    version = cfg.version

    print("source_dir ", source_dir)
    print("cfg.dataset.raw_data_dir: ", cfg.dataset.raw_data_dir)
    advattack = cfg.advType
    # Run model evaluation for WGAN/Autoencoder models
    perf_tst_df = pd.DataFrame([])
    perf_noi_df = pd.DataFrame([])
    perf_adv_df = pd.DataFrame([])
    perf_drp_df = pd.DataFrame([])
    
    # Running for different window size
    for window in cfg.windows:
        # print("Window size:", window)
        cfg.window = window
        dataset_dict = load_data_create_images(cfg) if not cfg.results_only else {}
        model_param = construct_model_cfg(cfg)
        adv_example_dict = {}

        # Run load and train for every model
        for model_indx, model_cfg in enumerate(model_param[0:]): #FIXME : Starting only with the first model
            model_type = model_cfg.model_type
            num_hid_layers = model_cfg.num_hid_layers
            noise_dim = model_cfg.noise_dim
            max_epoch = model_cfg.max_epoch
            model_id = '_'.join(str(val) for val in list(model_cfg.values())[1:])
            model_id = f"{model_cfg.model_type}_{window}_{model_id}"

            print("model_id :", model_id)   
            if model_type != 'wgan':
                continue
            print("Getting model..")
            wgan, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
            if remaining_epoch > 0:
                print("Model not found")
                continue
            model = wgan.discriminator

            # Getting the adversarial attacks for a single model
            if model_indx == 0:
                print(f"Generating FGSM Attacks")
                percent = 99.90
                epsilon = 0.02
                attack = "RandomPositionOffset"
                X_train = dataset_dict["No Attack"]["x_data"] 
                X_test = dataset_dict[attack]["x_data"][0:1000]
                y_test = dataset_dict[attack]["y_data"].astype(int)[0:1000]
                target_indices, colors, opt_factor = get_random_index(y_test, advattack) #get_random_index(dataset_dict, attack, advattack)
                model.compile(optimizer='adam',
                            loss= 'binary_crossentropy', #FIXME: BCE or MSE?
                            metrics=['accuracy'])
                # Run FGSM
                X_adv = X_test.copy()
                X_noi = X_test.copy()
                for indx in tqdm(target_indices[0:]):
                    x_target = tf.convert_to_tensor(X_test[indx].reshape((1,10,12,1)))
                    X_adv[indx], X_noi[indx] = fgsm_attack(model, x_target, opt_factor, epsilon)
            
            print(f"-------->>>> Model ID: {model_id} <<<<--------")
            p_ths = get_threshold(model, X_train, percent)
            perf_tst = get_performance(model_id, model, X_test, y_test, attack, p_ths)
            perf_noi = get_performance(model_id, model, X_noi, y_test, attack, p_ths)
            perf_adv = get_performance(model_id, model, X_adv, y_test, attack, p_ths)

            if perf_tst["fscore"] > 0.00:
                perf_tst_df.loc[noise_dim, num_hid_layers] = perf_tst["fscore"]
                perf_noi_df.loc[noise_dim, num_hid_layers] = perf_noi["fscore"]
                perf_adv_df.loc[noise_dim, num_hid_layers] = perf_adv["fscore"]
                perf_drp_df.loc[noise_dim, num_hid_layers] = perf_tst["fscore"] - perf_adv["fscore"]
            
        # Directories...
        data_dir = cfg.workspace_dir / 'artifacts' / f'results_{version}' / 'advAttacks' / f'adv_performance_drop_{cfg.window}.csv'
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        perf_drp_df.to_csv(data_dir)
        print(perf_drp_df)
        
   
# Main function
if __name__ == '__main__':
    adv_evaluate_pipeline()

"""
To run:

python run_adversarial_attacks.py version=december_dummy dataset=testing dataset.run_type=unit fast_load=True device=cpu
nohup python run_adversarial_attacks.py version=december_dummy dataset=testing dataset.run_type=unit fast_load=True device=cpu >/dev/null 2>&1 &
"""