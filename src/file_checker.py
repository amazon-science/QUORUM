import os
import numpy as np
import warnings

FOLDERS = ['claude', 'qwen', 'nova_pro']
BASE_PATH = 'summaries'

dataset_shapes = {}
dataset_presence = {}

for folder in FOLDERS:
    folder_path = os.path.join(BASE_PATH, folder)

    for filename in os.listdir(folder_path):
        if "all_predictions_" not in filename:
            continue

        dataset_name = filename.replace(".npy", "").replace("all_predictions_", "")
        dataset_presence.setdefault(dataset_name, set()).add(folder)

all_datasets = set(dataset_presence.keys())

for folder in FOLDERS:
    print(f"------ {folder} ------")
    folder_path = os.path.join(BASE_PATH, folder)

    present_datasets = set()

    for filename in os.listdir(folder_path):
        if "all_predictions_" not in filename:
            continue

        pred_path = os.path.join(folder_path, filename)
        conf_path = pred_path.replace("predictions", "confidences")

        predictions = np.load(pred_path)
        confidences = np.load(conf_path)

        dataset_name = filename.replace(".npy", "").replace("all_predictions_", "")
        current_shape = predictions.shape[0]

        present_datasets.add(dataset_name)

        if current_shape != confidences.shape[0]:
            warnings.warn(f"Shapes of dataset {dataset_name} do not correspond.")

        if dataset_name in dataset_shapes:
            if dataset_shapes[dataset_name] != current_shape:
                message = f"""Inconsistent shape for dataset '{dataset_name}': 
                {dataset_shapes[dataset_name]} (previous) vs {current_shape} (in {folder})"""
                warnings.warn(message)
        else:
            dataset_shapes[dataset_name] = current_shape

        print(f"Predictions: {dataset_name} --> {current_shape}")

    missing = all_datasets - present_datasets
    for dataset_name in missing:
        message = f"Dataset '{dataset_name}' missing in folder '{folder}'"
        warnings.warn(message)