import sys
# Make sure to adjust this path to your specific directory structure
sys.path.append("/vol/biomedic3/bglocker/mscproj/ms4824/stress-testing-framework/src")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.metrics import (
    roc_auc_score, recall_score, accuracy_score,
    confusion_matrix, balanced_accuracy_score, roc_curve
)
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

# This should now correctly import from your project structure
# Ensure you are importing the updated EmbedDataModule from the Canvas
from datasets.data import EmbedDataModule

# --- Model Wrapper for PRE-LOADED Models ---
class FeatureModelWrapper(nn.Module):
    """
    Wraps a pre-loaded, standard torchvision model to provide the .get_features()
    and .classify_features() methods required by the inference pipeline, without
    altering the underlying model's structure or state dict keys.
    """
    def __init__(self, preloaded_model: nn.Module, model_type: str):
        super().__init__()
        self.model_type = model_type
        self._model = preloaded_model

        # Define how to get the feature extractor and classifier from the preloaded model
        if model_type == "resnet":
            self._feature_extractor_module = nn.Sequential(*list(self._model.children())[:-1])
            self._classifier_module = self._model.fc
        elif model_type == "densenet":
            self._feature_extractor_module = self._model.features
            self._classifier_module = self._model.classifier
        elif model_type == "vit":
            # For ViT, the full model (sans head) is the feature extractor.
            self._feature_extractor_module = self._model
            # The classifier is the head of the ViT model
            self._classifier_module = self._model.heads.head
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported.")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts feature vectors from the input."""
        if self.model_type == "vit":
            # For ViT, we get features by temporarily replacing the head.
            original_head = self._classifier_module
            self._model.heads.head = nn.Identity()
            features = self._feature_extractor_module(x)
            self._model.heads.head = original_head # Restore the head
            return features

        # For ResNet and DenseNet
        feats = self._feature_extractor_module(x)
        if self.model_type == 'densenet':
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
        
        return torch.flatten(feats, 1)

    def classify_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Classifies features into logits."""
        return self._classifier_module(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines a standard forward pass for compatibility."""
        features = self.get_features(x)
        logits = self.classify_features(features)
        return logits

# --- Inference Function ---
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int
) -> Dict:
    """
    Main inference loop. Returns a dictionary with logits, probas, targets,
    predictions, features, and raw softmax confidence.
    """
    if not isinstance(model, FeatureModelWrapper):
        raise TypeError("The model passed to run_inference must be a FeatureModelWrapper instance.")

    model = model.eval().to(device)
    all_probas, all_targets, all_feats, all_logits = [], [], [], []

    for batch in tqdm(dataloader, desc="Inference"):
        with torch.no_grad():
            data = batch["x"].to(device)
            target = batch["y"]

            feats = model.get_features(data)
            logits = model.classify_features(feats)
            
            probas = torch.softmax(logits, 1)
            all_probas.append(probas.cpu())
            all_targets.append(target)
            all_feats.append(feats.cpu())
            all_logits.append(logits.cpu())
    
    model = model.cpu()
    
    if not all_targets:
        return {
            "logits": torch.empty(0, num_classes), "probas": torch.empty(0, num_classes),
            "predictions": torch.empty(0), "targets": torch.empty(0),
            "feats": np.empty(0), "softmax_confidence": torch.empty(0)
        }

    all_probas_cat = torch.cat(all_probas)
    results = {
        "logits": torch.cat(all_logits),
        "probas": all_probas_cat,
        "predictions": torch.argmax(all_probas_cat, 1),
        "targets": torch.cat(all_targets),
        "feats": torch.cat(all_feats).numpy(),
        "softmax_confidence": torch.max(all_probas_cat, 1)[0],
    }
    return results

def multiclass_specificity_score(y_true, y_pred, n_classes):
    """Calculates the average specificity for a multiclass classification problem."""
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specificities = []
    for i in range(n_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        specificities.append(specificity)
    return np.mean(specificities)

def calculate_metrics(results: Dict, num_classes: int) -> Dict:
    targets, preds, probas = results["targets"], results["predictions"], results["probas"]
    if len(targets) == 0:
        # Add new metrics to the list of NaNs for empty results
        return {k: np.nan for k in ['accuracy', 'auc', 'recall', 'specificity', 'balanced_accuracy', 'tpr_at_10_fpr', 'tpr_at_20_fpr']}
    
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'auc': roc_auc_score(targets, probas, multi_class='ovr', average='macro') if len(np.unique(targets)) > 1 else np.nan,
        'recall': recall_score(targets, preds, average='macro', zero_division=0),
        'specificity': multiclass_specificity_score(targets, preds, num_classes),
        'balanced_accuracy': balanced_accuracy_score(targets, preds),
    }

    tpr_at_10_fpr_list = []
    tpr_at_20_fpr_list = []

    # Calculate metrics only if there is more than one class present in the targets
    if len(np.unique(targets)) > 1:
        for i in range(num_classes):
            # Set up a 1-vs-all scenario for the current class
            y_true_class = (targets == i).numpy()
            y_score_class = probas[:, i].numpy()

            # Skip if a class is not present in the data subset (cannot compute ROC curve)
            if np.sum(y_true_class) == 0 or np.sum(y_true_class) == len(y_true_class):
                tpr_at_10_fpr_list.append(np.nan)
                tpr_at_20_fpr_list.append(np.nan)
                continue
            
            fpr, tpr, _ = roc_curve(y_true_class, y_score_class)
            
            # Find the TPR at the desired FPRs using linear interpolation
            tpr_at_10_fpr_list.append(np.interp(0.10, fpr, tpr))
            tpr_at_20_fpr_list.append(np.interp(0.20, fpr, tpr))
    
    # Add the mean TPR values to the metrics dictionary, ignoring NaNs
    metrics['tpr_at_10_fpr'] = np.nanmean(tpr_at_10_fpr_list) if tpr_at_10_fpr_list else np.nan
    metrics['tpr_at_20_fpr'] = np.nanmean(tpr_at_20_fpr_list) if tpr_at_20_fpr_list else np.nan
    
    return metrics

def main():
    # --- Configuration ---
    IMAGE_DIR = Path("/vol/biodata/data/Mammo/EMBED/images/png/1024x768")
    CSV_FILE = Path("embed_csv.csv")
    BATCH_SIZE = 64
    CHECKPOINT_DIR = "embed_model_checkpoints_patient_split"
    RESULTS_FILE = "25_label_shift_embed_stress_testing.csv"
    NUM_CLASSES = 4
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    
    models_to_run = [
        ("resnet", "resnet_trained_on_0_split_on_scanner_model_seed_33_epoch_15_10000_more_augments"),
        ("densenet", "densenet_trained_on_0_split_on_scanner_model_seed_33_epoch_15_10000_more_augments"),
        ("vit", "vit_trained_on_0_split_on_scanner_model_seed_33_epoch_15_10000_more_augments"),

        ("resnet", "resnet_trained_on_0_split_on_scanner_model_seed_34_epoch_15_10000_more_augments"),
        ("densenet", "densenet_trained_on_0_split_on_scanner_model_seed_34_epoch_15_10000_more_augments"),
        ("vit", "vit_trained_on_0_split_on_scanner_model_seed_34_epoch_15_10000_more_augments"),

        ("resnet", "resnet_trained_on_0_split_on_scanner_model_seed_35_epoch_15_10000_more_augments"),
        ("densenet", "densenet_trained_on_0_split_on_scanner_model_seed_35_epoch_15_10000_more_augments"),
        ("vit", "vit_trained_on_0_split_on_scanner_model_seed_35_epoch_15_10000_more_augments"),
    ]
    

    for model_t, model_name in models_to_run:
        print(f"\n\n--- Evaluating Model: {model_t.upper()} ({model_name}) ---")
        
        data_module = EmbedDataModule(csv_file=CSV_FILE, image_dir=IMAGE_DIR, batch_size=BATCH_SIZE, transforms=data_transforms)
        data_module.setup()
        
        # --- Programmatically define all 12 dataloaders to evaluate ---
        dataloaders_to_evaluate = {
            'test_a': data_module.test_dataloader_a(),
            'shifted_pos': data_module.test_dataloader_a_remove_pos(),
            'shifted_neg': data_module.test_dataloader_a_remove_neg(),
        }
        cf_scanners = [2, 3, 4]
        base_sets = {'a': 'a', 'a_remove_pos': 'shifted_pos', 'a_remove_neg': 'shifted_neg'}
        for base_key, out_prefix in base_sets.items():
            for scanner_id in cf_scanners:
                loader_name = f'cf_{out_prefix}_s{scanner_id}'
                method_name = f'cf_dataloader_{base_key}_s{scanner_id}'
                dataloaders_to_evaluate[loader_name] = getattr(data_module, method_name)()
        for scanner_id in cf_scanners:
            # Add the main real data dataloader
            loader_name = f'real_s{scanner_id}'
            method_name = f'test_dataloader_real_s{scanner_id}'
            if hasattr(data_module, method_name):
                dataloaders_to_evaluate[loader_name] = getattr(data_module, method_name)()

            # Add the positive-shifted real data dataloader
            loader_name_pos = f'real_s{scanner_id}_pos'
            method_name_pos = f'test_dataloader_real_s{scanner_id}_pos'
            if hasattr(data_module, method_name_pos):
                 dataloaders_to_evaluate[loader_name_pos] = getattr(data_module, method_name_pos)()

            # Add the negative-shifted real data dataloader
            loader_name_neg = f'real_s{scanner_id}_neg'
            method_name_neg = f'test_dataloader_real_s{scanner_id}_neg'
            if hasattr(data_module, method_name_neg):
                dataloaders_to_evaluate[loader_name_neg] = getattr(data_module, method_name_neg)()

        base_model = None
        if model_t == "resnet":
            base_model = models.resnet50(weights=None)
            base_model.fc = nn.Linear(base_model.fc.in_features, NUM_CLASSES)
        elif model_t == "densenet":
            base_model = models.densenet121(weights=None)
            base_model.classifier = nn.Linear(base_model.classifier.in_features, NUM_CLASSES)
        elif model_t == "vit":
            base_model = models.vit_b_16(weights=None)
            base_model.heads.head = nn.Linear(base_model.heads.head.in_features, NUM_CLASSES)
        
        CHECKPOINT_TO_LOAD = Path(CHECKPOINT_DIR) / f"{model_name}.pth"
        if not os.path.exists(CHECKPOINT_TO_LOAD):
            print(f"--- WARNING: Checkpoint file not found, skipping. Path: {CHECKPOINT_TO_LOAD} ---\n")
            continue
        
        print(f"--- Loading Checkpoint: {CHECKPOINT_TO_LOAD} ---")
        state_dict = torch.load(CHECKPOINT_TO_LOAD, map_location=device)
        base_model.load_state_dict(state_dict)
        print("--- Checkpoint Loaded Successfully ---")
        
        model = FeatureModelWrapper(base_model, model_t)
        
        # --- Run Inference and Calculate Metrics ---
        result_row = {"model_type": model_t, "model_name": model_name}
        for name, loader in dataloaders_to_evaluate.items():
            print(f"--- Processing dataset: {name} ---")
            inference_results = run_inference(model, loader, device,num_classes=NUM_CLASSES)
            metrics = calculate_metrics(inference_results, NUM_CLASSES)
            for metric_name, value in metrics.items():
                result_row[f"{name}_{metric_name}"] = value
        
        # Append the fully populated row to the list and save immediately
        results_df = pd.DataFrame([result_row])
        file_exists = os.path.isfile(RESULTS_FILE)
        results_df.to_csv(RESULTS_FILE, mode='a', header=not file_exists, index=False)
        print(f"Results for {model_name} appended successfully to {RESULTS_FILE}.")

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()
