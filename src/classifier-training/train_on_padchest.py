
import sys
sys.path.append("/vol/biomedic3/bglocker/mscproj/ms4824/stress-testing-framework/src")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from pathlib import Path


# --- Helper Functions & Classes ---
from helpers.helpers import set_seed
from datasets.data import PadChestDataModule

def evaluate_model(model, dataloader, criterion, device, phase_name="Eval"):
    model.eval()
    running_loss, all_labels, all_probs = 0.0, [], []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            
            inputs = batch['x'].to(device)
            labels = batch['y'].to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1] 
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            

    if not all_labels:
        return {'loss': 0, 'accuracy': 0, 'auc': 0}

    all_preds = (np.array(all_probs) > 0.5).astype(int)
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    
    return {'loss': epoch_loss, 'accuracy': epoch_acc, 'auc': epoch_auc}


# --- Main Execution Block ---
def main():
    # --- Configuration ---
    SEED = 33
    MODEL_TYPE = "resnet"
    SPLIT_ON = "ScannerType"
    SPLIT_VALUES = ("Imaging","Phillips")


    CSV_FILE = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
    IMAGE_DIR = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST/images")
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    CHECKPOINT_DIR = "model_checkpoints_patient_split"
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # --- Transformations ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5, translate=(0.15, 0.15), scale=(0.9, 1.1)), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
          
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Setup DataModule and Get Dataloaders ---
    data_module = PadChestDataModule(
        csv_file=CSV_FILE,
        image_dir=IMAGE_DIR,
        batch_size=BATCH_SIZE,
        transforms=data_transforms
    )
    data_module.setup(split_on=SPLIT_ON,split_values=SPLIT_VALUES)
    
    dataloaders = {
        'train': data_module.train_dataloader(),
        'val': data_module.val_dataloader(),
        'ood_val': data_module.ood_val_dataloader(),
        'test_a': data_module.test_dataloader_a(),
        'test_b': data_module.test_dataloader_b()
    }

    # --- Model, Criterion, Optimizer ---

    print(f"Model Type : {MODEL_TYPE}")
       
    if MODEL_TYPE == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features,2)
    if MODEL_TYPE == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    if MODEL_TYPE == "densenet":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT) 
        model.classifier = nn.Linear(model.classifier.in_features, 2)
   
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Training & Evaluation ---
    # Phase 1: Train head
    print("\n--- Phase 1: Training the classifier head ---")
    for param in model.parameters(): param.requires_grad = False

    if MODEL_TYPE == "resnet":
        for param in model.fc.parameters(): param.requires_grad = True
        optimizer_head = optim.AdamW(model.fc.parameters(), lr=1e-4)
    if MODEL_TYPE == "vit":
        for param in model.heads.parameters(): param.requires_grad = True
        optimizer_head = optim.AdamW(model.heads.parameters(), lr=1e-4)
    if MODEL_TYPE == "densenet":
        for param in model.classifier.parameters(): param.requires_grad = True
        optimizer_head = optim.AdamW(model.classifier.parameters(), lr=1e-4)
        
    
    
    for epoch in range(5):
        print(f'\nEpoch {epoch+1}/5 (Head Training)')
        model.train()
        
        # Initialize metric accumulators for training set
        running_loss, all_labels, all_probs = 0.0, [], []

        # Training loop without progress bar
        for batch in dataloaders['train']:
            if batch is None: continue
            inputs, labels = batch['x'].to(device), batch['y'].to(device)
            optimizer_head.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_head.step()

            # Accumulate metrics
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            
        # Calculate and print training metrics for the epoch
        if all_labels:
            train_preds = (np.array(all_probs) > 0.5).astype(int)
            train_loss = running_loss / len(all_labels)
            train_acc = accuracy_score(all_labels, train_preds)
            train_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
            print(f"{'Train':<8} | Loss: {train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f}")

        # Evaluate on validation set
        val_metrics = evaluate_model(model, dataloaders['val'], criterion, device, phase_name="Val")
        print(f"{'Val':<8} | Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['auc']:.4f}")

    # Phase 2: Fine-tune all
    print("\n--- Phase 2: Fine-tuning the entire network ---")
    for param in model.parameters(): param.requires_grad = True
    optimizer_full = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_full, mode='max', factor=0.1, patience=5)
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} (Full Fine-tuning)')
        model.train()

        # Initialize metric accumulators for training set
        running_loss, all_labels, all_probs = 0.0, [], []

        # Training loop without progress bar
        for batch in dataloaders['train']:
            if batch is None: continue
            inputs, labels = batch['x'].to(device), batch['y'].to(device)
            optimizer_full.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_full.step()

            # Accumulate metrics
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())

        # Calculate and print training metrics for the epoch
        if all_labels:
            train_preds = (np.array(all_probs) > 0.5).astype(int)
            train_loss = running_loss / len(all_labels)
            train_acc = accuracy_score(all_labels, train_preds)
            train_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
            print(f"{'Train':<8} | Loss: {train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f}")

        # Evaluate on validation sets
        val_metrics = evaluate_model(model, dataloaders['val'], criterion, device, phase_name="Val")
        print(f"{'Val':<8} | Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['auc']:.4f}")
        ood_val_metrics = evaluate_model(model, dataloaders['ood_val'], criterion, device, phase_name="OOD Val")
        print(f"{'OOD Val':<8} | Loss: {ood_val_metrics['loss']:.4f} Acc: {ood_val_metrics['accuracy']:.4f} AUC: {ood_val_metrics['auc']:.4f}")
        scheduler.step(val_metrics['auc'])
   
        save_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_TYPE}_trained_on_{SPLIT_VALUES[0]}_split_on_{SPLIT_ON}_seed_{SEED}_epoch_{epoch+1}_less_augments.pth") 
        torch.save(model.state_dict(), save_path)
 
        print(f"Epoch {epoch+1} | Model saved to {save_path}")

    print("\n--- Training Complete ---")
    
    # --- Final Evaluation ---
    print("\n--- Final Model Evaluation on Test Sets ---")
    test_a_metrics = evaluate_model(model, dataloaders['test_a'], criterion, device, phase_name="Test (A)")
    print(f"Test (Scanner A - In-Distribution) | Loss: {test_a_metrics['loss']:.4f} Acc: {test_a_metrics['accuracy']:.4f} AUC: {test_a_metrics['auc']:.4f}")
    test_b_metrics = evaluate_model(model, dataloaders['test_b'], criterion, device, phase_name="Test (B)")
    print(f"Test (Scanner B - OOD)            | Loss: {test_b_metrics['loss']:.4f} Acc: {test_b_metrics['accuracy']:.4f} AUC: {test_b_metrics['auc']:.4f}")

if __name__ == "__main__":
    main()