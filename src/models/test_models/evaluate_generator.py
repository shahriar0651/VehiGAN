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
import torch.nn.functional as F
_ = torch.manual_seed(123)
from helper import *



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


def evaluate_generator(cfg, model_cfg, wgan, dataset_dict, gen_file_name) -> dict:

    x_data = dataset_dict['No Attack']['x_data']

    sample_size= cfg.models.gen_sample_size
    noise_dim = model_cfg.noise_dim
    
    w_distance = 0
    kid_score = 0
    fid_score = 0
    repeat = 1
    verbose = False

    for i in range(repeat):
        # Real image samples
        indices = np.random.choice(x_data.shape[0], sample_size, replace=False)
        real_images = x_data[indices]
        # Fake image samples
        random_latent_vectors = tf.random.normal(shape=(sample_size, noise_dim))
        fake_images = wgan.generator(random_latent_vectors, training=False).numpy()
        
        target_size = (3, 299, 299)

        real_images = torch.from_numpy(real_images).permute(0, 3, 1, 2)
        fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2)
        
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
    
    gen_scores["W_Distance"] = w_distance/repeat
    gen_scores["FID_Score"] = fid_score/repeat
    gen_scores["KID_Score"] = kid_score/repeat
    print("gen_scores:", gen_scores)
    

    # Store dict data..
    output_file_path = Path(f"{gen_file_name}_dict.json")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w') as fp:
        fp.write(json.dumps(gen_scores, indent = 4))

    return gen_scores
