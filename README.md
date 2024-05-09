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

### Download the dataset

Download the dataset folder from (https://drive.google.com/drive/folders/1qISfmRi-9rz1uMHVM5la2DwvqN93DQGu?usp=sharing). 
Copy it just outside of the vehigan workspace, i.e., in '../datasets/VASP/'

## Set up VehiGAN Environment
### Clone VehiGAN and Create Environment
```
git clone https://github.com/shahriar0651/VehiGAN.git
cd VehiGAN
```
### Create Environment
```
mamba env create --file dependency/environment.yaml
mamba activate vehigan
```

If you already have created the environment, to update:
```
mamba env update --file dependency/environment.yaml --prune
mamba activate vehigan
```


# Train and Test VehiGAN

## Curate BSM Traces into MBDS dataset
```
cd src
python run_data_curation_pipeline.py -m dataset=training,testing
```

## Train Different Models (WGAN, and Baselines)
```
python run_model_training_pipeline.py -m models=wgan,autoencoder,baseline dataset=training dataset.run_type=unit
```

## Evaluate Individual Models (WGAN, and Baselines)
```
python run_ind_evaluation_pipeline.py -m models=wgan,autoencoder,baseline dataset=testing dataset.run_type=unit version=dummy
```
## Evaluate VehiGAN (Ensemble Models with WGAN)
```
python run_ens_evaluation_pipeline.py models=wgan dataset=testing dataset.run_type=unit version=dummy
```

## Visualize the results
- Run the `visualize_results.ipynb` notebook to generate graphs and show the results


## Folder Tree Structure


```
├── datasets
│   └── VASP
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
        ├── run_data_curation_pipeline.py
        ├── run_ens_evaluation_pipeline.py
        ├── run_ind_evaluation_pipeline.py
        ├── run_model_training_pipeline.py
        └── visualize_results.ipynb
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