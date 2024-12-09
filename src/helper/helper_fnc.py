from omegaconf import OmegaConf
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import joblib
import uuid
import copy

from models.get_models import get_ind_model


def construct_model_cfg(cfg:any)->list:
    model_param = []
    print("cfg.models: ", cfg.models)
    model_configs = cfg.models
    model_type = model_configs["model_type"]          
        
    if model_type == 'autoencoder':
        for num_hid_layers in model_configs.list_of_num_hid_layers:
            for max_epoch in model_configs.list_of_max_epoch:
                model_cfg = {
                    "model_type" : 'autoencoder',
                    "num_hid_layers": num_hid_layers,
                    'max_epoch' : max_epoch
                    }
                model_param.append(OmegaConf.create(model_cfg))

    
    elif model_type =='wgan':
        for num_hid_layers in model_configs.list_of_num_hid_layers:
            for noise_dim in model_configs.list_of_noise_dim:
                for max_epoch in model_configs.list_of_max_epoch:
                    model_cfg = {
                        "model_type" : 'wgan',
                        "num_hid_layers": num_hid_layers,
                        "noise_dim": noise_dim,
                        'max_epoch' : max_epoch,
                        'gen_sample_size' : model_configs.gen_sample_size
                        }
                    model_param.append(OmegaConf.create(model_cfg))
    return model_param


# Adversarial attacks
def get_representative_samples(cfg, dataset_dict, random = True):
    X_test = np.array([])  # Initialize as an empty array
    y_test = np.array([])  # Initialize as an empty array
    X_train = None  # Initialize as None
    y_train = None  # Initialize as None

    # Set the seed for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    for attack, data in tqdm(dataset_dict.items()):
        X_test_att = data["x_data"]
        y_test_att = data["y_data"].astype(int)
        
        if attack == "No Attack":
            X_train = X_test_att.copy()
            y_train = y_test_att.copy()
            continue
        if random == True:
            rndm_indx = np.random.choice(X_test_att.shape[0], 
                                        size=cfg.advSamples, 
                                        replace=False)
        else:
            rndm_indx = np.arange(0,cfg.advSamples)
        
        if X_test.size == 0:  
            X_test = X_test_att[rndm_indx]
            y_test = y_test_att[rndm_indx]
        else:
            X_test = np.concatenate((X_test, X_test_att[rndm_indx]), axis=0)
            y_test = np.concatenate((y_test, y_test_att[rndm_indx]), axis=0)
        
        print("X_train.shape : ", X_train.shape, "X_test.shape : ", X_test.shape)
    return X_train, y_train, X_test, y_test

def get_random_index(cfg, y_test, random=True):
    seed_value = 42
    np.random.seed(seed_value)
    if cfg.advType == 'fn':
        target_indices = np.where(y_test == 1)[0]   
        colors = ['red', 'green']    
        opt_factor = - 1  #FIXME: opt_factor =  1
    if cfg.advType == 'fp':
        target_indices = np.where(y_test == 0)[0] 
        colors = ['green', 'red'] 
        opt_factor =  1   #FIXME: opt_factor = -1
    if random==True:
        np.random.shuffle(target_indices)
    return target_indices, colors, opt_factor

def fgsm_attack(cfg, model, image, factor, epsilon, save = False):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction =  - model(image)
        loss =  factor * tf.reduce_mean(prediction)
    gradient = tape.gradient(loss, image)
    perturbation = epsilon * tf.sign(gradient)
    adv_image = image + perturbation

    # print("gradient:\n", gradient.numpy(), "\n\n")
    # print("image: \n", image.numpy(), "\n\n")
    # print("perturbation: \n", perturbation.numpy(), "\n\n")
    # print("adv_image: \n", adv_image.numpy(), "\n\n")

    # Generate random prefix
    if save:
        random_prefix = str(uuid.uuid4())
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_GAN/{random_prefix}_gradient.txt', np.squeeze(gradient.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_GAN/{random_prefix}_orgimage.txt', np.squeeze(image.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_GAN/{random_prefix}_perturbation.txt', np.squeeze(perturbation.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_GAN/{random_prefix}_advimage.txt', np.squeeze(adv_image.numpy()))
    return adv_image.numpy()

def pgd_attack(cfg, model, image, factor, epsilon, num_steps=25, step_size=0.001, save = False):
    # adv_image = image.copy()
    adv_image = copy.deepcopy(image)
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = - model(adv_image)
            loss = factor * tf.reduce_mean(prediction)
        gradient = tape.gradient(loss, adv_image)
        adv_image = tf.clip_by_value(adv_image + step_size * tf.sign(gradient), image - epsilon, image + epsilon)
    
    perturbation = adv_image - image
    
    if save:
        random_prefix = str(uuid.uuid4())
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_gradient.txt', np.squeeze(gradient.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_orgimage.txt', np.squeeze(image.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_perturbation.txt', np.squeeze(perturbation.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_advimage.txt', np.squeeze(adv_image.numpy()))

    return adv_image.numpy()

# def bim_attack(model, image, factor, epsilon, num_steps, step_size):
#     adv_image = image.copy()
#     for _ in range(num_steps):
#         with tf.GradientTape() as tape:
#             tape.watch(adv_image)
#             prediction = - model(adv_image)
#             loss = factor * tf.reduce_mean(prediction)
#         gradient = tape.gradient(loss, adv_image)
#         adv_image = tf.clip_by_value(adv_image + step_size * tf.sign(gradient), image - epsilon, image + epsilon)
#     return adv_image.numpy()


def fgsm_attack_ae(cfg, autoencoder, image, factor, epsilon, save = False):
    factor = tf.constant(factor, dtype=tf.float32)
    image = tf.cast(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = autoencoder(image)
        prediction = tf.cast(prediction, dtype=tf.float32)
        loss = factor * tf.norm((prediction-image), ord=2)
    gradient = tape.gradient(loss, image)
    perturbation = epsilon * tf.sign(gradient)
    adv_image = image + perturbation

    # print("gradient:\n", gradient.numpy(), "\n\n")
    # print("image: \n", image.numpy(), "\n\n")
    # print("perturbation: \n", perturbation.numpy(), "\n\n")
    # print("adv_image: \n", adv_image.numpy(), "\n\n")

    # Generate random prefix
    if save:
        random_prefix = str(uuid.uuid4())
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_AE/{random_prefix}_gradient.txt', np.squeeze(gradient.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_AE/{random_prefix}_orgimage.txt', np.squeeze(image.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_AE/{random_prefix}_perturbation.txt', np.squeeze(perturbation.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/FGSM_samples_AE/{random_prefix}_advimage.txt', np.squeeze(adv_image.numpy()))
    return adv_image.numpy()

def pgd_attack_ae(cfg, autoencoder, image, factor, epsilon, num_steps=25, step_size=0.001, save = False):
    # adv_image = image.copy()
    adv_image = copy.deepcopy(image)
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = autoencoder(image)
            prediction = tf.cast(prediction, dtype=tf.float32)
            loss = factor * tf.norm((prediction-image), ord=2)
        gradient = tape.gradient(loss, adv_image)
        adv_image = tf.clip_by_value(adv_image + step_size * tf.sign(gradient), image - epsilon, image + epsilon)
    
    perturbation = adv_image - image
    
    if save:
        random_prefix = str(uuid.uuid4())
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_gradient.txt', np.squeeze(gradient.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_orgimage.txt', np.squeeze(image.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_perturbation.txt', np.squeeze(perturbation.numpy()))
        np.savetxt(f'{cfg.workspace_dir}/artifacts/PGD_samples_GAN/{random_prefix}_advimage.txt', np.squeeze(adv_image.numpy()))

    return adv_image.numpy()



def fgsm_attack_multi(model_dict, image, factor, epsilon):
    with tf.GradientTape() as tape:
        loss = 0
        tape.watch(image)
        for _, model in model_dict.items():
            prediction =  - model(image)
            loss +=  factor * tf.reduce_mean(prediction)
    gradient = tape.gradient(loss, image)
    perturbation = epsilon * tf.sign(gradient)
    adv_image = image + perturbation
    return adv_image.numpy()

def pgd_attack_multi(model_dict, image, factor, epsilon, num_steps=25, step_size=0.001):
    # adv_image = image.copy()
    adv_image = copy.deepcopy(image)
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            loss = 0
            tape.watch(adv_image)
            for _, model in model_dict.items():
                prediction = - model(adv_image)
                loss +=  factor * tf.reduce_mean(prediction)
        gradient = tape.gradient(loss, adv_image)
        adv_image = tf.clip_by_value(adv_image + step_size * tf.sign(gradient), image - epsilon, image + epsilon)
    return adv_image.numpy()


def fgsm_attack_multi_ae(model_dict, image, factor, epsilon):
    with tf.GradientTape() as tape:
        loss = 0
        tape.watch(image)
        for _, model in model_dict.items():
            prediction = model(image)
            prediction = tf.cast(prediction, dtype=tf.float32)
            loss += factor * tf.norm((prediction-image), ord=2)
    gradient = tape.gradient(loss, image)
    perturbation = epsilon * tf.sign(gradient)
    adv_image = image + perturbation
    return adv_image.numpy()

def pgd_attack_multi_ae(model_dict, image, factor, epsilon, num_steps=25, step_size=0.001):
    adv_image = copy.deepcopy(image)
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            loss = 0
            tape.watch(adv_image)
            for _, model in model_dict.items():
                prediction = model(image)
                prediction = tf.cast(prediction, dtype=tf.float32)
                loss += factor * tf.norm((prediction-image), ord=2)
        gradient = tape.gradient(loss, adv_image)
        adv_image = tf.clip_by_value(adv_image + step_size * tf.sign(gradient), image - epsilon, image + epsilon)
    return adv_image.numpy()

def get_noisy_image(image, adv_image):
    seed_value = 42
    tf.random.set_seed(seed_value)
    perturbation = adv_image - image
    perturbation_flat = tf.reshape(perturbation, [-1])
    noise = tf.random.shuffle(perturbation_flat)
    noise = tf.reshape(noise, perturbation.shape)
    noisy_image = image + noise
    return noisy_image.numpy()


def get_threshold(cfg, model_id, model=None, X_train=None):
    print("Starting threshold.....")
    ext = f"{cfg.dataset.run_type}_{cfg.fast_load}_{model_id}"
    data_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'thresholds_{cfg.window}_{ext}.json'
    if Path(data_dir).exists():
        with open(data_dir, 'r') as fp:
            threshold_dict = json.load(fp)  
        print(f"\n\nThreshold loaded for {model_id}\n\n")
        return threshold_dict
    p_train = - model.predict(X_train)
    threshold_dict = {}
    for percent in np.arange(90, 100, 0.5):
        threshold_dict[f"{percent}"] = np.percentile(p_train, percent)
    with open(data_dir, 'w') as fp:
        fp.write(json.dumps(threshold_dict, indent=4))   
    print(f"\n\nThreshold saved for {model_id}\n\n")

    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    for scaler_name, scaler in scalers.items():
        scaler_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'scaler_{scaler_name}_{cfg.window}_{ext}.pkl'
        _ = scaler.fit_transform(p_train.reshape(-1, 1))
        joblib.dump(scaler, scaler_dir)
        print(f"{scaler_name} scaler saved!")
    return threshold_dict

def get_threshold_ae(cfg, model_id, model=None, X_train=None):
    print("Starting threshold.....")
    ext = f"{cfg.dataset.run_type}_{cfg.fast_load}_{model_id}"
    data_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'thresholds_ae_{cfg.window}_{ext}.json'
    if Path(data_dir).exists():
        with open(data_dir, 'r') as fp:
            threshold_dict = json.load(fp)  
        print(f"\n\nThreshold loaded for {model_id}\n\n")
        return threshold_dict
    recon_loss = model.predict(X_train)- X_train
    recon_loss = recon_loss.reshape(recon_loss.shape[0], -1)
    p_train = tf.norm(recon_loss, ord=2, axis=1).numpy()
    threshold_dict = {}
    for percent in np.arange(90, 100, 0.5):
        threshold_dict[f"{percent}"] = np.percentile(p_train, percent)
    with open(data_dir, 'w') as fp:
        fp.write(json.dumps(threshold_dict, indent=4))   
    print(f"\n\nThreshold saved for {model_id}\n\n")

    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    for scaler_name, scaler in scalers.items():
        scaler_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'scaler_ae_{scaler_name}_{cfg.window}_{ext}.pkl'
        _ = scaler.fit_transform(p_train.reshape(-1, 1))
        joblib.dump(scaler, scaler_dir)
        print(f"{scaler_name} scaler saved!")
    return threshold_dict

def load_scaler (cfg, model_id, scaler_name):
    # Scale the data
    ext = f"{cfg.dataset.run_type}_{cfg.fast_load}_{model_id}"
    scaler_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'scaler_ae_{scaler_name}_{cfg.window}_{ext}.pkl'
    scaler = joblib.load(scaler_dir)
    return scaler

def calculate_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)
    TNR = tn / (tn + fp)
    FNR = fn / (fn + tp)
    return TPR, FPR, TNR, FNR

def performance_metrics(y_test, y_pred):
    print(f"Shape: {y_test.shape}, {y_pred.shape}")
    pre, rec, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    tpr, fpr, tnr, tnr = calculate_rates(y_test, y_pred)
    performance = {
        "precision": pre,
        "recall": rec,
        "fscore": fscore,
        "support": support,
        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "FNR": tnr
    }  
    # print(f"{model_id} : ", performance)
    return performance

def get_performance(model_id, model, X_test, y_test, p_ths):
    prediction = - model.predict(X_test).flatten()
    y_pred = (prediction > p_ths).astype(int)
    performance = performance_metrics(y_test, y_pred)
    return performance 

def get_performance_ae(model_id, model, X_test, y_test, p_ths):
    # prediction = - model.predict(X_test).flatten()
    # prediction = model(X_test)
    # prediction = tf.norm(prediction-prediction, ord=2,  axis=(1, 2, 3)).numpy()
    recon_loss = model.predict(X_test)- X_test
    recon_loss = recon_loss.reshape(recon_loss.shape[0], -1)
    prediction = tf.norm(recon_loss, ord=2, axis=1).numpy()

    y_pred = (prediction > p_ths).astype(int)
    performance = performance_metrics(y_test, y_pred)
    return performance 
    # pre, rec, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    # tpr, fpr, tnr, tnr = calculate_rates(y_test, y_pred)
    # performance = {
    #     "precision": pre,
    #     "recall": rec,
    #     "fscore": fscore,
    #     "support": support,
    #     "TPR": tpr,
    #     "FPR": fpr,
    #     "TNR": tnr,
    #     "FNR": tnr
    # }  
    # # print(f"{model_id} : ", performance)
    # return performance


def save_adversarial_data(cfg, advCap_model, model_id, X_adv, y_test):
    ext = f"{cfg.advType}_{cfg.advFnc}_{cfg.advCap}_{cfg.epsilon}_{model_id}_{advCap_model}_{cfg.advSamples}"
    X_data_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'adv_X_data_{cfg.window}_{ext}.npy'
    y_data_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'adv_y_data_{cfg.window}_{ext}.npy'
    X_data_dir.parent.mkdir(parents=True, exist_ok=True)
    np.save(X_data_dir, X_adv)
    np.save(y_data_dir, y_test)
    print("X_adv Saved!")

def load_adversarial_data(cfg, target_model_id, advCap_model):
    ext = f"{cfg.advType}_{cfg.advFnc}_{cfg.advCap}_{cfg.epsilon}_{target_model_id}_{advCap_model}_{cfg.advSamples}"
    X_data_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'adv_X_data_{cfg.window}_{ext}.npy'
    y_data_dir = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / 'advAttacks' / f'adv_y_data_{cfg.window}_{ext}.npy'
    X_test = np.load(X_data_dir)
    y_test = np.load(y_data_dir)
    print("X_adv Loaded!")
    return X_test, y_test

def load_model(cfg, model_cfg):
    wgan, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
    if remaining_epoch > 0:
        print("Model is not trained yet!")
        return None
    model = wgan.discriminator
    model.compile(optimizer='adam',
        loss= 'binary_crossentropy', #FIXME: BCE or MSE?
        metrics=['accuracy'])
    return model

def get_prediction(cfg, 
                   model_id,
                   attack = None, 
                   evalType='benign',
                   model_cfg = None,
                   X_train=None,
                   X_test=None,
                   y_test=None,
                   ):
    model = load_model(cfg, model_cfg)
    if evalType=='benign':
        dis_file_name = cfg.workspace_dir / 'artifacts' / f'results_{cfg.version}' / f"dis_{model_id}"
        pred_data = pd.read_csv(f"{dis_file_name}_{attack}_pred.csv", index_col=0)
        pred_score = pred_data["prediction"].values.reshape(-1, 1)
        y_test_attack = pred_data["ground_truth"].astype(int)
    elif evalType=='adversarial':
        pred_score = - model.predict(X_test).flatten()
        y_test_attack = y_test
    
    p_ths = get_threshold(cfg, model_id, model, X_train)[f"{cfg.th_percent}"]
    # p_ths = get_threshold(cfg, model_id)[f"{cfg.th_percent}"]
    y_pred = (pred_score > p_ths).astype(int)
    return pred_score, y_test_attack, y_pred

def get_robust_ens_performance(cfg, wgan_eval_df_scld, 
                               attack = None, 
                               k_max = None, 
                               m_max = None, 
                               evalType=None,
                               model_cfg_dict = None,
                               X_train=None,
                               X_test=None,
                               y_test=None,
                               p_ths = None):
    
    result_dict = {metric: pd.DataFrame() for metric in cfg.metrics}
    pred_scores_all = pd.DataFrame([])
    pred_class_all = pd.DataFrame([])

    for model_count in tqdm(range(m_max)):
        model_id = wgan_eval_df_scld.index[model_count]
        model_cfg = model_cfg_dict[model_id]
        pred_score, y_test_attack, y_pred = get_prediction(cfg, 
                                                   model_id, 
                                                   attack = attack, 
                                                   evalType=evalType, 
                                                   model_cfg = model_cfg,
                                                   X_train=X_train,
                                                   X_test=X_test,
                                                   y_test=y_test,)
        
        # scaler = load_scaler(cfg, model_id, scaler_name=cfg.scaler) #TODO: Add loaded scaler 
        # pred_score = scaler.transform(pred_score.reshape(-1, 1)).flatten()
        scaler = StandardScaler()
        pred_score = scaler.fit_transform(pred_score.reshape(-1, 1)).flatten()
        pred_scores_all[model_id] = pred_score
        pred_class_all[model_id] = y_pred.flatten()

    print("pred_scores_all : ", pred_scores_all)
    print("Prediction Loading Complete!")

    n_samples = pred_scores_all.shape[0]
    
    for m in tqdm(range(1, m_max+1), position=0):
        k_max_temp = m 
        if k_max_temp > k_max:
            k_max_temp = k_max

        rng = np.random.default_rng()
        mask = np.ones((n_samples, m), dtype=int) * np.array(range(m))
        mask = rng.permuted(mask, axis=1)

        score_data = pd.DataFrame(
            pred_scores_all.values[np.arange(n_samples)[:, None], mask],
            # index=pred_class_all.index) #TODO : Trying max voting
            index=pred_scores_all.index) #TODO : Trying average voting
        cumulative_avg_df = score_data.expanding(axis=1).mean()

        class_data = pd.DataFrame(
            pred_class_all.values[np.arange(n_samples)[:, None], mask],
            index=pred_class_all.index)
        cumulative_class_df = class_data.expanding(axis=1).mean()

        for k in tqdm(range(1, k_max_temp+1), position=1, leave=False):
            y_score_attack = cumulative_avg_df[k-1]
            auroc = roc_auc_score(y_test_attack, y_score_attack)
            auprc = average_precision_score(y_test_attack, y_score_attack)
            y_class_attack = (cumulative_class_df[k-1] > 0.5).astype(int)
            pre, rec, fscore, support = precision_recall_fscore_support(y_test_attack, y_class_attack, average='binary')
            tpr, fpr, tnr, fnr = calculate_rates(y_test_attack, y_class_attack)
            result_dict['auroc'].loc[k, m] = auroc
            result_dict['auprc'].loc[k, m] = auprc
            result_dict['fpr'].loc[k, m] = fpr
            result_dict['fnr'].loc[k, m] = fnr
            result_dict['pre'].loc[k, m] = pre
            result_dict['rec'].loc[k, m] = rec
            result_dict['fscore'].loc[k, m] = fscore
            
    return result_dict

def get_robust_ens_performance_multi(cfg, 
                                    wgan_eval_df_scld, 
                                    attack = None, 
                                    k_max = None, 
                                    m_max = None, 
                                    evalType=None,
                                    model_cfg_dict = None,
                                    X_train=None,
                                    ):
                            #    X_test=None,
                            #    y_test=None,
                            #    p_ths = None,
                        
    
    result_dict = {metric: pd.DataFrame() for metric in cfg.metrics}
    pred_scores_all = pd.DataFrame([])
    pred_class_all = pd.DataFrame([])

    for m in tqdm(range(1, m_max+1), position=0):
        model_id = wgan_eval_df_scld.index[m-1]
        pred_scores_all = pd.DataFrame([])
        pred_class_all = pd.DataFrame([])
        scaler_dict = {}
        X_test, y_test = load_adversarial_data(cfg, model_id, advCap_model = 'White-box')
        for model_count in tqdm(range(m)):
            model_id = wgan_eval_df_scld.index[model_count]
            model_cfg = model_cfg_dict[model_id]
            pred_score, y_test_attack, y_pred = get_prediction(cfg, 
                                                    model_id, 
                                                    attack = attack, 
                                                    evalType=evalType, 
                                                    model_cfg = model_cfg,
                                                    X_train=X_train,
                                                    X_test=X_test,
                                                    y_test=y_test,)
            
            # scaler = load_scaler(cfg, model_id, scaler_name=cfg.scaler) #TODO: Add loaded scaler 
            # pred_score = scaler.transform(pred_score.reshape(-1, 1)).flatten()
            scaler = StandardScaler()
            pred_score = scaler.fit_transform(pred_score.reshape(-1, 1)).flatten()

            pred_scores_all[model_id] = pred_score
            pred_class_all[model_id] = y_pred
        print("pred_scores_all : ", pred_scores_all.head(5))
        print("Prediction Loading Complete!")
        n_samples = pred_scores_all.shape[0]

        print("m : ", m)
        k_max_temp = m 
        if k_max_temp > k_max:
            k_max_temp = k_max

        rng = np.random.default_rng()
        mask = np.ones((n_samples, m), dtype=int) * np.array(range(m))
        mask = rng.permuted(mask, axis=1)

        score_data = pd.DataFrame(
            pred_scores_all.values[np.arange(n_samples)[:, None], mask],
            index=pred_scores_all.index)
        cumulative_avg_df = score_data.expanding(axis=1).mean()

        class_data = pd.DataFrame(
            pred_class_all.values[np.arange(n_samples)[:, None], mask],
            index=pred_class_all.index)
        cumulative_class_df = class_data.expanding(axis=1).mean()

        for k in tqdm(range(1, k_max_temp+1), position=1, leave=False):
            y_score_attack = cumulative_avg_df[k-1]
            auroc = roc_auc_score(y_test_attack, y_score_attack)
            auprc = average_precision_score(y_test_attack, y_score_attack)
            y_class_attack = (cumulative_class_df[k-1] > 0.5).astype(int)
            pre, rec, fscore, support = precision_recall_fscore_support(y_test_attack, y_class_attack, average='binary')
            tpr, fpr, tnr, fnr = calculate_rates(y_test_attack, y_class_attack)
            result_dict['auroc'].loc[k, m] = auroc
            result_dict['auprc'].loc[k, m] = auprc
            result_dict['fpr'].loc[k, m] = fpr
            result_dict['fnr'].loc[k, m] = fnr
            result_dict['pre'].loc[k, m] = pre
            result_dict['rec'].loc[k, m] = rec
            result_dict['fscore'].loc[k, m] = fscore
    return result_dict


