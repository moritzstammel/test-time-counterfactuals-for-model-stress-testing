import sys
sys.path.append("/vol/biomedic3/bglocker/mscproj/ms4824/stress-testing-framework/src")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import (
    roc_auc_score, recall_score, accuracy_score,
    confusion_matrix, balanced_accuracy_score, roc_curve
)
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

# Make sure this import points to the location of your updated datamodule file
from datasets.data import PadChestDataModule

# --- Model Wrapper for PRE-LOADED Models ---
class FeatureModelWrapper(nn.Module):
    """
    Wraps a pre-loaded, standard torchvision model to provide the .get_features()
    and .classify_features() methods required by the inference pipeline.
    """
    def __init__(self, preloaded_model: nn.Module, model_type: str):
        super().__init__()
        self.model_type = model_type
        self._model = preloaded_model
        if model_type == "resnet":
            self._feature_extractor_module = nn.Sequential(*list(self._model.children())[:-1])
            self._classifier_module = self._model.fc
        elif model_type == "densenet":
            self._feature_extractor_module = self._model.features
            self._classifier_module = self._model.classifier
        elif model_type == "vit":
            # For ViT, we extract features before the final classification head
            self._feature_extractor_module = self._model
            self._classifier_module = self._model.heads.head
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported.")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "vit":
            # Temporarily replace head with identity to get features
            original_head = self._classifier_module
            self._model.heads.head = nn.Identity()
            features = self._feature_extractor_module(x)
            self._model.heads.head = original_head # Restore it
            return features
        
        feats = self._feature_extractor_module(x)
        if self.model_type == 'densenet':
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
        return torch.flatten(feats, 1)

    def classify_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self._classifier_module(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        logits = self.classify_features(features)
        return logits

# --- Inference and Metric Calculation Functions ---
def run_inference(model: torch.nn.Module, dataloader: DataLoader, device: str) -> Dict:
    """
    Main inference loop. Returns a dictionary with logits, probas, targets, and predictions.
    """
    if dataloader is None:
        return {"probas": torch.empty(0), "predictions": torch.empty(0), "targets": torch.empty(0)}

    model = model.eval().to(device)
    all_probas, all_targets = [], []

    for batch in tqdm(dataloader, desc="Inference"):
        if batch is None: continue
        with torch.no_grad():
            data, target = batch["x"].to(device), batch["y"]
            feats = model.get_features(data)
            logits = model.classify_features(feats)
            probas = torch.softmax(logits, 1)
            all_probas.append(probas.cpu())
            all_targets.append(target.cpu())
    
    model.cpu()
    
    if not all_targets:
        return {"probas": torch.empty(0), "predictions": torch.empty(0), "targets": torch.empty(0)}

    all_probas_cat = torch.cat(all_probas)
    return {
        "probas": all_probas_cat,
        "predictions": torch.argmax(all_probas_cat, 1),
        "targets": torch.cat(all_targets),
    }

def calculate_metrics(results: Dict) -> Dict:
    """Calculates all required metrics from an inference results dictionary."""
    targets = results["targets"]
    predictions = results["predictions"]
    probas = results["probas"]

    if len(targets) == 0:
        return {
            'accuracy': np.nan, 'auc': np.nan, 'recall': np.nan,
            'specificity': np.nan, 'balanced_accuracy': np.nan,
            'tpr_at_10%_fpr': np.nan, 'tpr_at_20%_fpr': np.nan
        }
    
    auc = roc_auc_score(targets, probas[:, 1]) if len(np.unique(targets)) > 1 else 0.5
    fpr, tpr, _ = roc_curve(targets, probas[:, 1]) if len(np.unique(targets)) > 1 else ([0], [0], [])
    
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = (cm.ravel() if len(cm.ravel()) == 4 else (cm[0,0], 0, 0, 0) if cm.shape == (1,1) else (0,0,0,0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': accuracy_score(targets, predictions),
        'auc': auc,
        'recall': recall_score(targets, predictions, zero_division=0),
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy_score(targets, predictions),
        'tpr_at_10%_fpr': np.interp(0.10, fpr, tpr),
        'tpr_at_20%_fpr':  np.interp(0.20, fpr, tpr)
    }

def main():
    # --- Configuration ---
    CSV_FILE = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
    IMAGE_DIR = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST/images")
    BATCH_SIZE = 64
    CHECKPOINT_DIR = "model_checkpoints_patient_split"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define the models you want to test
    models_to_test = [
          ("resnet", "resnet50_io_trained_on_female15"),
            ("densenet", "densenet121_io_trained_on_female11"),
            ("vit", "vit_b_16_io_trained_on_female5"),

            ("resnet", "resnet_trained_on_F_split_on_Sex_seed_34_epoch_15"),
            ("densenet", "densenet_trained_on_F_split_on_Sex_seed_34_epoch_15"),
            ("vit", "vit_trained_on_F_split_on_Sex_seed_34_epoch_15"),

            ("resnet", "resnet_trained_on_F_split_on_Sex_seed_35_epoch_15"),
            ("densenet", "densenet_trained_on_F_split_on_Sex_seed_35_epoch_15"),
            ("vit", "vit_trained_on_F_split_on_Sex_seed_35_epoch_15"),

    ]
    
    RESULTS_FILE = "000_baseline_stress_test_results_female.csv"
    all_results_list = []

    for model_t, model_name in models_to_test:
        print(f"\n===== Evaluating Model: {model_t.upper()} ({model_name}) =====")

        # --- MODIFICATION: Use the new setup method ---
        data_module = PadChestDataModule(csv_file=CSV_FILE, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE)
        data_module.setup_baseline_stresstesting(split_on= "Sex", split_values= ("F", "M") )
        
        # --- MODIFICATION: Get the dictionary of dataloaders ---
        dataloaders_to_evaluate = data_module.get_stress_test_dataloaders()

        # --- Model Loading ---
        base_model = None
        if model_t == "resnet":
            base_model = models.resnet50(weights=None)
            base_model.fc = nn.Linear(base_model.fc.in_features, 2)
        elif model_t == "densenet":
            base_model = models.densenet121(weights=None)
            base_model.classifier = nn.Linear(base_model.classifier.in_features, 2)
        elif model_t == "vit":
            base_model = models.vit_b_16(weights=None)
            base_model.heads.head = nn.Linear(base_model.heads.head.in_features, 2)
        
        CHECKPOINT_TO_LOAD = Path(CHECKPOINT_DIR) / f"{model_name}.pth"
        if not os.path.exists(CHECKPOINT_TO_LOAD):
            print(f"--- WARNING: Checkpoint not found, skipping. Path: {CHECKPOINT_TO_LOAD} ---\n")
            continue
        
        state_dict = torch.load(CHECKPOINT_TO_LOAD, map_location=device)
        base_model.load_state_dict(state_dict)
        model = FeatureModelWrapper(base_model, model_t)
        model.to(device)

        # --- MODIFICATION: Loop over the dictionary of dataloaders ---
        result_row = {"model_type": model_t, "model_name": model_name}
        
        for name, loader in dataloaders_to_evaluate.items():
            print(f"--- Processing dataset: {name} ---")
            if loader is None:
                print("Skipping empty dataloader.")
                continue
            
            inference_results = run_inference(model, loader, device)
            metrics = calculate_metrics(inference_results)
            
            for metric_name, value in metrics.items():
                result_row[f"{name}_{metric_name}"] = value
        
        all_results_list.append(result_row)

    # --- Save all collected results to a single CSV file at the end ---
    if all_results_list:
        print(f"\nSaving all results to {RESULTS_FILE}...")
        final_df = pd.DataFrame(all_results_list)
        final_df.to_csv(RESULTS_FILE, index=False)
        print("Evaluation finished and results saved successfully.")
    else:
        print("\nNo results were generated to save.")

if __name__ == "__main__":
    main()
