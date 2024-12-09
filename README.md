# VehiGAN
Code and dataset for the paper:
**VehiGAN: Generative Adversarial Networks for Adversarially Robust V2X Misbehavior Detection Systems**

## Pre-requisite
### System Requirements
The following implementation is only for the following (changes may be needed for other systems):
- Ubuntu 20.04 
- Python 3.10
- Nvidia GPU

### Install Mamba or Miniconda
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
chmod +x Mambaforge-Linux-x86_64.sh
./Mambaforge-Linux-x86_64.sh
```


## Set up VehiGAN Environment

### Clone VehiGAN and Create Environment
----
```bash
git clone https://github.com/shahriar0651/VehiGAN.git VehiGAN
cd VehiGAN
```
### Create Environment
```bash
conda env create --file dependency/environment.yaml
conda activate vehigan
```

If you already have created the environment, to update:
```bash
mamba env update --file dependency/environment.yaml --prune
mamba activate vehigan
```

----

### Download the *MisbehaviorX* dataset

Download the dataset folder from [IEEE Data Port](https://ieee-dataport.org/documents/misbehaviorx-comprehensive-v2x-misbehavior-detection-dataset-enabled-v2x-application#) Or from a temporary [google drive folder](https://drive.google.com/drive/folders/1qISfmRi-9rz1uMHVM5la2DwvqN93DQGu?usp=sharing). 

Place it just outside of the VehiGAN workspace as follows.

```
├── datasets
│   └── MisbehaviorX
│       ├── ambients
│       └── attacks
└── VehiGAN
```
----

# **Train and Test VehiGAN**

**Note**: To process the complete dataset, use the parameter `dataset.run_type=full` in the commands or for unit testing use `dataset.run_type=unit fast_load=True`.

---

## **Step 1: Curate BSM Traces into MBDS Dataset**
Navigate to the `src` directory and execute the data curation pipeline to process the training and testing datasets. This step organizes raw data into the MBDS format.

```bash
cd src
python step_1_run_data_curation_pipeline.py -m dataset=training,testing
```

---

## **Step 2: Train Different Models (WGAN, Autoencoder, and Baselines)**
Run the training pipeline for various models, specifying the model types (`wgan`, `autoencoder`, `baseline`) and limiting the dataset to a unit sample (`dataset.run_type=unit`) for testing purposes. 

```bash
python step_2_run_ind_training_pipeline.py -m models=wgan,autoencoder,baseline dataset=training dataset.run_type=unit fast_load=True
```

---

## **Step 3: Evaluate Individual Models (WGAN, Autoencoder, and Baselines)**
Evaluate each trained model on the testing dataset. 

```bash
python step_3_run_ind_detect_evaluation.py -m models=wgan,autoencoder,baseline dataset=testing dataset.run_type=unit fast_load=True
```

---

## **Step 4: Evaluate VehiGAN (Ensemble Models with WGAN)**  
- First, load and merge the individual WGAN model performances (from step 3) and save them in CSV format.  
- Then, select the top `m` models based on their scores (Generator/Discriminator) and evaluate the detection performance for the ensemble model.  

```bash
python step_4_run_ens_fixed_detect_evaluation.py models=wgan dataset=testing dataset.run_type=unit fast_load=True
```

---

## **Step 5: Evaluate Adversarial Robustness of VehiGAN (Ensemble Models with WGAN)**  
Generate adversarial samples to evaluate the robustness of individual models against benign, adversarial, and noisy datasets. The parameters include:  
- `advCap`: specifies the type of evaluation (`indv` for individual models).  
- `advFnc`: the adversarial attack function (`fgsm` in this case).  
- `epsilon`: the perturbation level for adversarial samples.  

```bash
python step_5_run_ind_robustness_evaluation.py -m dataset=testing dataset.run_type=unit fast_load=True advCap=indv evalType=adversarial m_max=5 advRandom=True epsilon=0.00,0.005,0.010,0.015,0.020 advFnc='fgsm'
```

---

## **Step 6: Evaluate Benign and Adversarial Robustness of VehiGAN (Ensemble Models with WGAN)**  
### **Evaluate on Benign Datasets**  
Test the ensemble model (using `m_max` and `k_max` parameters) on benign datasets.

```bash
python step_6_run_ens_random_robust_evaluation.py -m dataset=testing dataset.run_type=unit fast_load=True evalType=benign
```

### **Evaluate on Adversarial Datasets**  
Evaluate the ensemble model's robustness on adversarial datasets using various perturbation levels (`epsilon`) and attack capabilities (`advCap`).  

```bash
python step_6_run_ens_random_robust_evaluation.py -m dataset=testing dataset.run_type=unit fast_load=True evalType=adversarial advCap=indv,trans,multi epsilon=0.00,0.005,0.010,0.015,0.020
```

---

## **Visualize the Results**
To visualize and analyze the results, open and run the `step_final_visualize_results.ipynb` notebook. This generates graphs and provides a summary of the experiment outcomes.

---

## Folder Tree Structure


```
├── datasets
│   └── MisbehaviorX
│       ├── ambients
│       └── attacks
└── VehiGAN
    ├── artifacts
    ├── config
    │   ├── config.yaml
    │   ├── dataset
    │   └── models
    ├── dependency
    │   └── environment.yaml
    ├── docs
    ├── README.md
    ├── references
    ├── results
    └── src
        ├── dataset
        ├── helper
        ├── models
        ├── step_1_run_data_curation_pipeline.py
        ├── step_2_run_ind_training_pipeline.py
        ├── step_3_run_ind_detect_evaluation.py
        ├── step_4_run_ens_fixed_detect_evaluation.py
        ├── step_5_run_adv_robust_evaluation_pipeline_gan.py
        ├── step_6_run_ens_robust_evaluation_pipeline_adv.py
        └── step_final_visualize_results.ipynb
```


## Citation

```bibtex
@inproceedings{shahriar2024vehigan,
  title={VehiGAN: Generative Adversarial Networks for Adversarially Robust V2X Misbehavior Detection Systems},
  author={Shahriar, Md Hasan and Ansari, Mohammad Raashid and Monteuuis, Jean-Philippe and Chen, Cong and Petit, Jonathan and Hou, Y. Thomas and Lou, Wenjing},
  booktitle={The 44th IEEE International Conference on Distributed Computing Systems (ICDCS)},
  year={2024}
}
```

## Acknowledgments
- This project relies on [VASP](https://github.com/quic/vasp) for V2X simulation and attack data generation.
- This project relies on [keras.io/examples](https://keras.io/examples/generative/wgan_gp/) for GAN implementation.

## Funding

This work was supported in part by the US National Science Foundation under grants 1837519, 2235232 and 2312447, and by the Office of Naval Research under grant N00014-19-1-2621.