from omegaconf import OmegaConf
from omegaconf import DictConfig
from omegaconf import open_dict
import json

import numpy as np
from numpy import asarray, cov, iscomplexobj, trace
from skimage.transform import resize
from scipy.linalg import sqrtm
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

import torch
import numpy as np
from scipy.stats import wasserstein_distance
from torch.autograd import Variable
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
import numpy as np
import torch
from scipy.stats import gaussian_kde

import torch.nn.functional as F
_ = torch.manual_seed(123)
from helper import *
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
import numpy as np
from scipy.linalg import sqrtm

import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from scipy.stats import norm
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tsfel
import time

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.callbacks import EarlyStopping


def get_w_distance(real_images, fake_images, verbose=False):
    # Initialize histograms
    hist1 = np.zeros((256,), dtype=np.float32)
    hist2 = np.zeros((256,), dtype=np.float32)

    # Compute histograms for dataset 1
    for image in real_images:
        hist, _ = np.histogram(image.ravel(), 256, [0, 1])
        hist1 += hist.flatten()

    # Compute histograms for dataset 2
    for image in fake_images:
        hist, _ = np.histogram(image.ravel(), 256, [0, 1])
        hist2 += hist.flatten()

    # Normalize histograms
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    emd = wasserstein_distance(hist1, hist2)
    if verbose:
        print("emd: ", emd)
        print("emd: ", type(emd))
    return emd


def get_fid_score(real_images, fake_iamges, verbose=False, feature=64):
    fid = FrechetInceptionDistance(normalize=True, feature=feature)
    fid.update(real_images, real=True)
    fid.update(fake_iamges, real=False)
    fid_score = fid.compute().numpy().item()
    if verbose:
        print("fid_score: ", fid_score)
        print("fid_score: ", type(fid_score))

    return fid_score


def get_kid_score(real_images, fake_iamges, verbose=False, feature=64, subset_size=50):
    kid = KernelInceptionDistance(normalize=True, feature=feature, subset_size=subset_size)
    kid.update(real_images, real=True)
    kid.update(fake_iamges, real=False)
    kid_score, _ = kid.compute()
    kid_score = kid_score.numpy().item()
    if verbose:
        print("kid_score: ", kid_score)
        print("kid_score: ", type(kid_score))

    return kid_score


def get_mifid_score(real_images, fake_iamges, verbose=False, feature=64):
    mifid = MemorizationInformedFrechetInceptionDistance(normalize=True, feature=feature)
    mifid.update(real_images, real=True)
    mifid.update(fake_iamges, real=False)
    mifid_score = mifid.compute().numpy().item()
    if verbose:
        print("mifid: ", mifid_score)
        print("mifid: ", type(mifid_score))

    return mifid_score

# KDE Divergence with random sampling for high-dimensional data
def kde_divergence(X_real_reshaped, X_fake_reshaped, num_samples=1000):
    # Convert PyTorch tensors to NumPy arrays
    # X_real_np = X_real.detach().cpu().numpy()
    # X_fake_np = X_fake.detach().cpu().numpy()
    
    # # Reshape from (n, H, W) to (n*H, W)
    _, W = X_real_reshaped.shape
    # X_real_reshaped = X_real_np.reshape(-1, W)
    # X_fake_reshaped = X_fake_np.reshape(-1, W)
    
    # Fit KDE for real and generated data (each column corresponds to a variable)
    kde_real = gaussian_kde(X_real_reshaped.T)
    kde_gen = gaussian_kde(X_fake_reshaped.T)
    
    # Randomly sample points from the combined range of both datasets
    min_val = min(X_real_reshaped.min(), X_fake_reshaped.min())
    max_val = max(X_real_reshaped.max(), X_fake_reshaped.max())
    
    # Generate random samples in the 10-dimensional space
    sample_points = np.random.uniform(min_val, max_val, size=(W, num_samples))
    
    # Evaluate KDEs at the random sample points
    p_real = kde_real(sample_points)
    p_gen = kde_gen(sample_points)
    
    # Compute L2 distance (squared difference) between density estimates
    divergence = np.sum((p_real - p_gen) ** 2)
    
    return divergence

# def get_vif_score(real_images, fake_iamges, verbose=False):
#     vif = VisualInformationFidelity()
#     vif_score = vif(real_images, fake_iamges).numpy().item()
#     if verbose:
#         print("vif score: ", vif_score)
#         print("vif_score: ", type(vif_score))
#     return vif_score


# def get_lpips_score(real_images, fake_iamges, verbose=False):
#     lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="squeeze")
#     lpips_score = lpips(torch.clamp(real_images, 0, 1), torch.clamp(fake_iamges, 0, 1)).detach().numpy().item()
#     if verbose:
#         print("lpips score: ", lpips_score)
#         print("lpips: ", type(lpips))
#     return lpips_score


def mmd(X_real, X_fake, kernel_bandwidth=1.0):
    # Compute the RBF kernel (Gaussian) between datasets
    K_real_real = rbf_kernel(X_real, X_real, gamma=1.0 / (2 * kernel_bandwidth ** 2))
    K_fake_fake = rbf_kernel(X_fake, X_fake, gamma=1.0 / (2 * kernel_bandwidth ** 2))
    K_real_fake = rbf_kernel(X_real, X_fake, gamma=1.0 / (2 * kernel_bandwidth ** 2))
    
    # Calculate MMD
    mmd_value = np.mean(K_real_real) + np.mean(K_fake_fake) - 2 * np.mean(K_real_fake)
    
    return mmd_value


def pca_similarity(X_real, X_fake, n_components=2):
    # Apply PCA to both datasets
    pca = PCA(n_components=n_components)
    X_real_pca = pca.fit_transform(X_real)
    X_fake_pca = pca.fit_transform(X_fake)
    
    # Compute Euclidean distance between the mean of the PCA projections
    mean_real = np.mean(X_real_pca, axis=0)
    mean_fake = np.mean(X_fake_pca, axis=0)
    
    return euclidean_distances([mean_real], [mean_fake])[0][0]

def frechet_distance(X_real, X_fake):
    # Compute mean and covariance of real data
    mu_real = np.mean(X_real, axis=0)
    sigma_real = np.cov(X_real, rowvar=False)
    
    # Compute mean and covariance of fake data
    mu_fake = np.mean(X_fake, axis=0)
    sigma_fake = np.cov(X_fake, rowvar=False)
    
    # Compute the Frechet Distance
    mu_diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    
    # Handle numerical instability in sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    frechet_dist = np.sum(mu_diff**2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return frechet_dist


def ks_test_multidimensional(X_real, X_fake):
    print("Staring KS test........")
    num_features = X_real.shape[1]
    print(X_real.shape, X_fake.shape)
    ks_statistics = []
    p_values = []
    
    # Perform KS test for each feature individually
    for i in tqdm(range(num_features), desc= "Working on Feature:"):
        # print("Working on Feature: ", i+1)
        try:
            ks_stat, p_value = ks_2samp(X_real[:, i], X_fake[:, i])
        except TypeError:
            print("Skipping as the feature contains None type value")
            continue
        ks_statistics.append(ks_stat)
        p_values.append(p_value)

    # Aggregate the results (mean of KS statistics)
    mean_ks_stat = np.mean(ks_statistics)
    mean_p_values = np.mean(p_values)
    return mean_ks_stat

def ad_test_multidimensional(X_real, X_fake):
    num_features = X_real.shape[1]
    ad_statistics = []
    p_values = []
    
    # Perform Anderson-Darling test for each feature individually
    for i in range(num_features):
        ad_stat, critical_values, p_value = anderson_ksamp([X_real[:, i], X_fake[:, i]])
        ad_statistics.append(ad_stat)
        p_values.append(p_value)
    
    # Aggregate the results (mean of AD statistics)
    mean_ad_stat = np.mean(ad_statistics)
    return mean_ad_stat


def wasserstein_multidimensional(X_real, X_fake):
    num_features = X_real.shape[1]
    distances = []
    
    # Perform Wasserstein distance for each feature individually
    for i in range(num_features):
        distance = wasserstein_distance(X_real[:, i], X_fake[:, i])
        distances.append(distance)
    
    # Aggregate the results (mean of Wasserstein distances)
    mean_distance = np.mean(distances)
    return mean_distance

def bhattacharyya_distance(X_real, X_fake):
    num_features = X_real.shape[1]
    distances = []
    
    # Perform Bhattacharyya distance for each feature individually
    for i in range(num_features):
        # Estimate normal distribution parameters for real and fake data
        mu_real, sigma_real = np.mean(X_real[:, i]), np.std(X_real[:, i])
        mu_fake, sigma_fake = np.mean(X_fake[:, i]), np.std(X_fake[:, i])
        
        # Calculate Bhattacharyya distance
        dist = 0.25 * np.log(0.25 * (sigma_real**2 / sigma_fake**2 + sigma_fake**2 / sigma_real**2 + 2)) + \
               0.25 * (mu_real - mu_fake)**2 / (sigma_real**2 + sigma_fake**2)
        distances.append(dist)
    
    # Aggregate the results (mean Bhattacharyya distance)
    mean_distance = np.mean(distances)
    return mean_distance


def torch_to_numpy(tensor):
    if torch.is_tensor(tensor):  # Check if it's a PyTorch tensor
        if tensor.is_cuda:  # Check if it's on the GPU
            tensor = tensor.cpu()  # Move to CPU if it's on the GPU
        return tensor.detach().numpy()  # Convert to NumPy
    else:
        return tensor  # Return as is if it's not a tensor
    
def feature_similarity_score(X_real, X_fake, domain='temporal'):
    
    print("Starting feature_similarity_score :", domain)

    cgf_file = tsfel.get_features_by_domain(domain)

    # X_real = np.squeeze(torch_to_numpy(X_real))
    # X_fake = np.squeeze(torch_to_numpy(X_fake))

    # F_real = []
    # F_fake = []

    # X_real = X_real.reshape(-1, X_real.shape[-1])
    # X_fake = X_fake.reshape(-1, X_fake.shape[-1])

    F_real = tsfel.time_series_features_extractor(cgf_file, 
                                                    X_real, 
                                                    fs=50, 
                                                    window_size=10,
                                                    verbose = 0).values  
    F_fake = tsfel.time_series_features_extractor(cgf_file, 
                                                    X_fake, 
                                                    fs=50, 
                                                    window_size=10,
                                                    verbose = 0).values 

    print("Feature extracted: ", F_real.shape, F_fake.shape)

    # # Function to check if a column contains only float elements
    # def is_column_float(column):
    #     try:
    #         # Convert the entire column to float
    #         column.astype(np.float64)
    #         return True
    #     except ValueError:
    #         return False

    # # # Identify columns that are entirely float for both F_real and F_fake
    # # F_real = np.vstack(F_real)    
    # # F_fake = np.vstack(F_fake)   

    # valid_columns = []
    # for i in range(F_real.shape[1]):
    #     if is_column_float(F_real[:, i]) and is_column_float(F_fake[:, i]):
    #         valid_columns.append(i)

    # print(f"Actual features: {F_real.shape[1]}")
    # # Select only valid columns for both F_real and F_fake
    # F_real = F_real[:, valid_columns]
    # F_fake = F_fake[:, valid_columns]

    # print(f"Selected features: {F_real.shape[1]}")
    # print(F_real)
    # print(F_fake)
    # similarity_score = similarity_function(F_real, F_fake)
    similarity_score = ks_test_multidimensional(F_real, F_fake)

    return similarity_score


def train_on_X_test_on_X(cfg, X_real, X_fake, eval_type='TSTR'):
    
    if eval_type == 'TRTS':
        X_train = X_real
        X_test = X_fake
    elif eval_type == 'TSTR':
        X_train = X_fake
        X_test = X_real

    X_train = np.transpose(X_train, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))

    # define an AE (CNN or RNN)
    window = 10 #cfg.window
    num_signals = 12 # cfg.num_signals
    max_epochs = 1000
    n_filters_list = [128, 64, 32]
    crop_factors = {}
    crop_factors [8] = ((4, 4), (2, 2))
    crop_factors [10] = ((3, 3), (2, 2))
    crop_factors [12] = ((2, 2), (2, 2))

    # Encoder
    input_img = keras.Input(shape=(window, num_signals, 1))  
    x = ZeroPadding2D(padding=2)(input_img)
    for n_filters in n_filters_list:    
        print(n_filters)
        x = Conv2D(n_filters, (2, 2), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(input_img, x)

    # Decoder
    encoded_img = keras.Input(shape= x.shape[1:])  
    x = encoded_img
    for n_filters in n_filters_list[::-1]:    
        x = Conv2D(n_filters, (2, 2), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = Cropping2D(cropping= crop_factors [window])(x)
    decoder = keras.Model(encoded_img, decoded)
    
    # Autoencoder
    x = keras.Input(shape=(window, num_signals, 1))
    autoencoder = keras.Model(x, decoder(encoder(x)))

    # Compile model
    opt = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.5, beta_2=0.99)
    autoencoder.compile(loss='mse', optimizer=opt, metrics=['mse'])
    autoencoder.summary()
    cbk = EarlyStopping(monitor='val_mse', mode="auto", verbose=1, patience=5)
    # train the model
    history = autoencoder.fit(X_train, X_train,
                    epochs=max_epochs,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    validation_data=(X_train, X_train,), callbacks=[cbk])

    # test the model
    X_test_recon = autoencoder.predict(X_test)
    # y_pred_attack = np.linalg.norm(X_test_recon - X_test, axis=(1, 2)).flatten()
    score = np.linalg.norm(X_test_recon - X_test)
    return score


def evaluate_generator(cfg, model_cfg, wgan, dataset_dict, gen_file_name) -> dict:

    x_data = dataset_dict['No Attack']['x_data']

    sample_size= cfg.models.gen_sample_size
    noise_dim = model_cfg.noise_dim
    
    w_distance = 0
    kid_score = 0
    fid_score = 0

    kde_score = 0
    
    mmd_score = 0
    fd_score = 0
    ks_score = 0
    ad_score = 0
    wmd_score = 0
    bd_score = 0
    fat_score = 0
    fas_score = 0
    trts_score = 0
    tstr_score = 0

    repeat = 10
    verbose = False


    for i in range(repeat):
        # Real image samples
        indices = np.random.choice(x_data.shape[0], sample_size, replace=False)
        real_images = x_data[indices]
        # Fake image samples
        random_latent_vectors = tf.random.normal(shape=(sample_size, noise_dim))
        fake_images = wgan.generator(random_latent_vectors, training=False).numpy()
        

        real_images = torch.from_numpy(real_images).permute(0, 3, 1, 2)
        fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2)

        X_real_np = real_images.detach().cpu().numpy()
        X_fake_np = fake_images.detach().cpu().numpy()
        
        # Reshape from (n, H, W) to (n*H, W)
        n, C, H, W = X_real_np.shape
        X_real_reshaped = X_real_np.reshape(-1, W)
        X_fake_reshaped = X_fake_np.reshape(-1, W)

        print(real_images.shape, fake_images.shape)
        print(X_real_reshaped.shape, X_fake_reshaped.shape)
        print("Rehshaped done!")
        # # Merge to create a multi-dimensional dataset
        # kde_score += kde_divergence(X_real_reshaped, X_fake_reshaped, num_samples=sample_size) 
        # mmd_score += mmd(X_real_reshaped, X_fake_reshaped)
        # fd_score += frechet_distance(X_real_reshaped, X_fake_reshaped)
        # ks_score += ks_test_multidimensional(X_real_reshaped, X_fake_reshaped)
        # ad_score += ad_test_multidimensional(X_real_reshaped, X_fake_reshaped)[0]
        # wmd_score += wasserstein_multidimensional(X_real_reshaped, X_fake_reshaped)
        # bd_score += bhattacharyya_distance(X_real_reshaped, X_fake_reshaped)
        # fat_score += feature_similarity_score(X_real_np, X_fake_np, ks_test_multidimensional, domain='temporal')
        # fas_score += feature_similarity_score(X_real_np, X_fake_np, ks_test_multidimensional, domain='statistical')


        # 1. KDE Divergence
        start_time = time.time()
        kde_score += kde_divergence(X_real_reshaped, X_fake_reshaped, num_samples=sample_size)
        end_time = time.time()
        print(f"KDE Divergence execution time: {end_time - start_time} seconds")

        # 2. MMD Score
        start_time = time.time()
        mmd_score += mmd(X_real_reshaped, X_fake_reshaped)
        end_time = time.time()
        print(f"MMD Score execution time: {end_time - start_time} seconds")

        # 3. Frechet Distance
        start_time = time.time()
        fd_score += frechet_distance(X_real_reshaped, X_fake_reshaped)
        end_time = time.time()
        print(f"Frechet Distance execution time: {end_time - start_time} seconds")

        # 4. KS Test (Multidimensional)
        start_time = time.time()
        ks_score += ks_test_multidimensional(X_real_reshaped, X_fake_reshaped)
        end_time = time.time()
        print(f"KS Test execution time: {end_time - start_time} seconds")

        # 5. AD Test (Multidimensional)
        start_time = time.time()
        ad_score += ad_test_multidimensional(X_real_reshaped, X_fake_reshaped)
        end_time = time.time()
        print(f"AD Test execution time: {end_time - start_time} seconds")

        # 6. Wasserstein Distance
        start_time = time.time()
        wmd_score += wasserstein_multidimensional(X_real_reshaped, X_fake_reshaped)
        end_time = time.time()
        print(f"Wasserstein Distance execution time: {end_time - start_time} seconds")

        # # 7. Bhattacharyya Distance
        # start_time = time.time()
        # bd_score += bhattacharyya_distance(X_real_reshaped, X_fake_reshaped)
        # end_time = time.time()
        # print(f"Bhattacharyya Distance execution time: {end_time - start_time} seconds")

        # 8. Feature Similarity Score (Temporal Domain)
        start_time = time.time()
        fat_score += feature_similarity_score(X_real_reshaped, X_fake_reshaped, domain='temporal')
        end_time = time.time()
        print(f"Feature Similarity Score (Temporal) execution time: {end_time - start_time} seconds")

        # 9. Feature Similarity Score (Statistical Domain)
        start_time = time.time()
        fas_score += feature_similarity_score(X_real_reshaped, X_fake_reshaped, domain='statistical')
        end_time = time.time()
        print(f"Feature Similarity Score (Statistical) execution time: {end_time - start_time} seconds")

        # 10. Starting Model Testing-based Eval
        start_time = time.time()
        trts_score += train_on_X_test_on_X(cfg, X_real_np, X_fake_np, eval_type='TRTS')
        end_time = time.time()
        print(f"train_on_X_test_on_X (TSTR) execution time: {end_time - start_time} seconds")

        # 11. Starting Model Testing-based Eval
        start_time = time.time()
        tstr_score += train_on_X_test_on_X(cfg, X_real_np, X_fake_np, eval_type='TSTR')
        end_time = time.time()
        print(f"train_on_X_test_on_X (TSTR) execution time: {end_time - start_time} seconds")

        # Gen-Score for Inception-based metric
        target_size = (3, 299, 299)
        real_images = F.interpolate(
            real_images, size=(target_size[1], target_size[2]), mode="bilinear", align_corners=False
        )
        real_images = torch.cat([real_images, real_images, real_images], dim=1)
        fake_images = F.interpolate(
            fake_images, size=(target_size[1], target_size[2]), mode="bilinear", align_corners=False
        )
        fake_images = torch.cat([fake_images, fake_images, fake_images], dim=1)
        print(real_images.shape, fake_images.shape)

        w_distance = w_distance + get_w_distance(real_images, fake_images, verbose=verbose)
        fid_score = fid_score + get_fid_score(real_images, fake_images, verbose=verbose)
        kid_score = kid_score + get_kid_score(real_images, fake_images, verbose=verbose)
        
    # Call each function and store the results in a dictionary
    gen_scores = {}
    
    gen_scores["SS_FAT_Score"] = fat_score / repeat
    gen_scores["SS_FAS_Score"] = fas_score / repeat
    gen_scores["SS_FAC_Score"] = (fat_score+fas_score) / 2 * repeat

    gen_scores["SS_TRTS_Score"] = trts_score / repeat
    gen_scores["SS_TSTR_Score"] = tstr_score / repeat
    gen_scores["SS_TXTX_Score"] = (trts_score+tstr_score) / 2 * repeat

    gen_scores["TS_MMD_Score"] = mmd_score / repeat
    gen_scores["TS_FD_Score"] = fd_score / repeat
    gen_scores["TS_KS_Score"] = ks_score / repeat
    gen_scores["TS_AD_Score"] = ad_score / repeat
    gen_scores["TS_WMD_Score"] = wmd_score / repeat
    gen_scores["TS_BD_Score"] = bd_score / repeat
    gen_scores["TS_KDE_Score"] = kde_score/repeat

    gen_scores["IM_W_Distance"] = w_distance/repeat
    gen_scores["IM_FID_Score"] = fid_score/repeat
    gen_scores["IM_KID_Score"] = kid_score/repeat

    
    mmd_score = 0
    fd_score = 0
    ks_score = 0
    ad_score = 0
    wmd_score = 0
    bd_score = 0


    print("gen_scores:", gen_scores)
    

    # Store dict data..
    output_file_path = Path(f"{gen_file_name}_dict.json")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w') as fp:
        fp.write(json.dumps(gen_scores, indent = 4))

    return gen_scores
