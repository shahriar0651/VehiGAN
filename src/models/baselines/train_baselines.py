from pathlib import Path
import time
from joblib import dump, load
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def train_all_baselines(cfg, baseline_model_dict, X_train):

    model_root_dir = cfg.models_dir
    model_type = cfg.models.model_type
    num_signals = len(cfg.features)


    training_time_dict = {}
    for model_name, model in baseline_model_dict.items():
        print("Model name: ", model_name)
        start_time = time.time()
        model.fit(X_train)
        training_time = time.time() - start_time
        training_time_dict[model_name] = training_time
        print("Training time: ", training_time)
        
        # save the model
        model_dir = Path(f'{model_root_dir}/{model_type}/{model_type}_{num_signals}_{model_name}')
        model_dir.parent.mkdir(exist_ok=True, parents=True)
        dump(model, model_dir)
        print(f"{model_name} trained and saved!")
        
    # Store dict data..
    file_dir = Path(f"../artifacts/models/baselines/training_time.json")
    file_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(file_dir, 'w') as fp:
        fp.write(json.dumps(training_time_dict, indent = 4))
    print(f"Training complete!")

def load_all_baselines(cfg):
    
    model_root_dir = cfg.models_dir
    model_type = cfg.models.model_type
    num_signals = len(cfg.features)

    baseline_model_dict = {}
    for model_name in cfg.models.model_List:
        model_dir = Path(f'{model_root_dir}/{model_type}/{model_type}_{num_signals}_{model_name}')
        baseline_model_dict[model_name] = load(model_dir)
        print(f"{model_name} loaded !")
    return baseline_model_dict
