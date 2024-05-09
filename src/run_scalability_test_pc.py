import os
from pathlib import Path

import hydra
import argparse
import yaml
from omegaconf import DictConfig

from dataset import *
from models import *
from helper import *
from keras.models import save_model, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten


def save_and_load_normal_model(cfg, model_id, model):
    model_dir = Path(f"{cfg.models_dir}/scalability_test/tfnorm_{model_id}.h5")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_dir)
    loaded_model = load_model(model_dir)
    return loaded_model

# Initialize loggers
def inference_time_normal(cfg, model_id, model, dummy_input):
    model = save_and_load_normal_model(cfg, model_id, model)
    timings = np.zeros((cfg.scale_repeat, 1))
    warmup_input = np.random.randn(1000, 10, 12, 1).astype(np.float32)
    _ = model.predict(warmup_input, verbose=False)
    # Measure performance
    for rep in range(cfg.scale_repeat):
        start_time = tf.timestamp()
        output = model.predict(dummy_input, verbose=False)
        end_time = tf.timestamp()
        curr_time = (end_time - start_time) * 1000
        timings[rep] = curr_time
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    print(f"Inference time for standard model: {mean_syn} +- {std_syn} ms")
    print(f"Output: {output}")
    return mean_syn, std_syn


def save_and_load_tflite_model(cfg, model_id, model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    model_dir = Path(f"{cfg.models_dir}/scalability_test/tflite_{model_id}.tflite")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(model_dir, "wb") as f:
        f.write(tflite_model)
    interpreter = tf.lite.Interpreter(model_path=str(model_dir))
    return interpreter


# Initialize loggers
def inference_time_tflite(cfg, model_id, model, dummy_input):
    interpreter = save_and_load_tflite_model(cfg, model_id, model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    timings = np.zeros((cfg.scale_repeat, 1))

    for rep in range(cfg.scale_repeat):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]["index"], [dummy_input[0]])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        end_time = tf.timestamp()
        curr_time = (end_time - start_time) * 1000
        timings[rep] = curr_time

    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    print(f"Inference time for tflite model: {mean_syn} +- {std_syn} ms")
    print(f"Output: {output}")
    return mean_syn, std_syn


@hydra.main(config_path="../config", config_name="config.yaml")
def model_scalability_pipeline(cfg: DictConfig) -> None:

    # TODO: Set memory grouth....
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print("gpus : ", gpus)
    usingGPU=False
    device='CPU'
    if gpus:
        usingGPU=True
        device='GPU'
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # ......................
    # Define directory
    source_dir = Path(__file__).resolve().parent
    cfg.workspace_dir = source_dir.parent
    cfg.models_dir = source_dir / cfg.models_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    cfg.dataset.raw_data_dir = source_dir / cfg.dataset.raw_data_dir
    cfg.dataset.clean_data_dir = source_dir / cfg.dataset.clean_data_dir
    if not cfg.models_dir.exists():
        os.makedirs(cfg.models_dir)
    if not cfg.scaler_dir.exists():
        os.makedirs(cfg.scaler_dir)

    # Run model training for WGAN or Autoencoder
    if cfg.models.model_type != "baselines":
        # Running for different window size
        for window in cfg.windows:
            print("Window size: ", window)
            cfg.window = window
            # Load training data
            model_param = construct_model_cfg(cfg)
            # Run load and train for every model
            inference_time_df = pd.DataFrame([])
            for model_id, model_cfg in enumerate(model_param):
                # model_id = "_".join(list(model_cfg.values()))
                model_id = "_".join(str(value) for value in model_cfg.values())
                model_type = model_cfg.model_type
                print(f"\n\nStrating: {model_type}_{model_id}")
                # Load model
                models, cbk, remaining_epoch = get_ind_model(cfg, model_cfg)
                # Train model
                if remaining_epoch > 0:
                    print(f"Training model for remaining {remaining_epoch} epochs")
                else:
                    print("Model is already trained.")

                # ------------------------
                model = models.discriminator
                # Set GPU memory growth to avoid allocating all GPU memory

                # Get the list of visible devices
                visible_devices = tf.config.experimental.list_physical_devices("GPU")
                if len(visible_devices) > 0:
                    print("Keras is ... using the GPU for inference.")
                else:
                    print("Keras is not using the GPU for inference.")

                dummy_input = np.random.randn(1, 10, 12, 1).astype(np.float32)
                mean_syn, _ = inference_time_normal(cfg, model_id, model, dummy_input)
                
                inference_time_model = pd.DataFrame([])
                inference_time_model['Model Type'] = [f'Standard ({device})']
                inference_time_model['Inference Time'] = [mean_syn]
                inference_time_model["Model"] = [model_id]
                inference_time_df = pd.concat([inference_time_df, inference_time_model], axis=0)

                if device == 'GPU':
                    continue
                mean_syn, _ = inference_time_tflite(cfg, model_id, model, dummy_input)
                inference_time_model = pd.DataFrame([])
                inference_time_model['Model Type'] = ['TFLite (CPU)']
                inference_time_model['Inference Time'] = [mean_syn]
                inference_time_model["Model"] = [model_id]
                inference_time_df = pd.concat([inference_time_df, inference_time_model], axis=0)

            # Plot the correlation heatmap..........
            plot_dir = Path(f"{source_dir}/../artifacts/plots/data_analysis//")
            plot_dir.mkdir(parents=True, exist_ok=True)
            inference_time_df.to_csv(f"{plot_dir}/scalability_inference_comp_{usingGPU}.csv")


# Main function
if __name__ == "__main__":
    model_scalability_pipeline()
