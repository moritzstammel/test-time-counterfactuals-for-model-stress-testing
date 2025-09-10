import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import cv2
from torchvision import transforms
from torchvision.transforms import functional as F

# --- Helper Functions & Constants ---

def log_broken_image(path):
    """Logs the path of a broken image."""
    print(f"Logging broken image: {path}")
    with open("broken_images_embed.txt", "a") as f:
        f.write(f"{path}\n")

MODELNAME_MAP = {
    "Selenia Dimensions": 0, "Senographe Pristina": 1,
    "Senograph 2000D ADS_17.4.5": 2, "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3, "Clearview CSm": 4,
    "Senographe Essential VERSION ADS_53.40": 5, "Senographe Essential VERSION ADS_54.10": 5
}
TISSUE_MAPS = {"A": 0, "B": 1, "C": 2, "D": 3}

def prepare_embed_csv(csv_file, image_root_dir):
    """Loads and prepares the master EMBED DataFrame."""
    print("Loading and preparing the master EMBED DataFrame...")
    df = pd.read_csv(csv_file)

    df["image_path"] = df["image_path"].apply(lambda x: Path(image_root_dir) / str(x))
    df["scanner_model"] = df.ManufacturerModelName.apply(lambda x: MODELNAME_MAP.get(x, -1))
    df["tissue_density"] = df.tissueden.apply(lambda x: TISSUE_MAPS.get(x, -1))
    df['view_position'] = df.ViewPosition.apply(lambda x: 0 if x == 'MLO' else 1)

    df = df.dropna(subset=["tissueden", "scanner_model", "view_position", "image_path"])
    df.rename(columns={"empi_anon_x": "PatientID"}, inplace=True, errors='ignore')
    df = df[df.FinalImageType == "2D"].copy()

    print("Master EMBED DataFrame prepared.")
    return df

# --- EMBED Dataset Class ---

class MammographyDataset(Dataset):
    """Custom PyTorch Dataset for EMBED mammography images."""
    def __init__(self, df, image_root_dir, transform=None, counterfactual_scanner: int = None):
        self.df = df
        self.image_root_dir = Path(image_root_dir)
        self.transform = transform
        self.counterfactual_scanner = counterfactual_scanner
        self.use_scanner_counterfactuals = (counterfactual_scanner is not None)
        self.counterfactual_dir = Path("/vol/biomedic3/bglocker/mscproj/ms4824/stress-testing-framework/src/datasets/counterfactual-images/EMBED/scanner")
        
        self.image_paths = df["image_path"].values
        self.patient_ids = df["PatientID"].values
        self.scanners = df["scanner_model"].values
        self.tissue_density_labels = df["tissue_density"].values

    def __len__(self):
        return len(self.df)

    def _read_image(self, idx):
        """Reads and processes a single mammogram image."""
        original_path = Path(self.image_paths[idx])
        if self.use_scanner_counterfactuals:
            relative_path = original_path.relative_to(self.image_root_dir)
            image_path = self.counterfactual_dir / str(self.counterfactual_scanner) / relative_path
        else:
            image_path = original_path

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Isolate the largest connected component (the breast tissue)
        thresh = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)[1]
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        if nb_components > 1:
            max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (output == max_label).astype(np.uint8)
            img_array = (image * mask)
        else:
            img_array = image
        
        # Normalize and convert to PIL for transforms
        max_val = img_array.max()
        if max_val > 0:
            img_array = img_array / max_val
        pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
        return pil_image.convert("RGB")

    def __getitem__(self, idx):
        try:
            label = getattr(self, f"{self.target_label}_labels")[idx]
            image = self._read_image(idx)
        except Exception as e:
            image_path = self.image_paths[idx]
            print(f"ERROR: Could not load image {image_path}. Error: {e}. Skipping.", flush=True)
            log_broken_image(image_path)
            return None
        
        image_tensor = self.transform(image) if self.transform else image
        
        return {
            'x': image_tensor,
            'y': torch.tensor(label, dtype=torch.long),
            'scanner': self.scanners[idx],
            'patient_id': self.patient_ids[idx]
        }

# --- EMBED DataModule Class ---

class EmbedDataModule:
    """Manages data loading and splitting for the EMBED dataset."""
    def __init__(self, csv_file, image_dir, batch_size, transforms, num_workers=6):
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.num_workers = num_workers

    def _get_data_splits(self, master_df):
        """Performs a stratified, patient-aware split of the data."""
        print("Performing patient-aware stratified splits...")
        y_stratify = master_df.groupby("PatientID")["tissueden"].first().values
        patient_ids = master_df.PatientID.unique()

        train_ids, val_test_ids = train_test_split(patient_ids, test_size=0.25, random_state=33, stratify=y_stratify)
        val_ids, test_ids = train_test_split(val_test_ids, test_size=(len(val_test_ids) - 600)/len(val_test_ids), random_state=33) # 600 for val
        
        return {
            "train": master_df[master_df.PatientID.isin(train_ids)],
            "val": master_df[master_df.PatientID.isin(val_ids)],
            "test": master_df[master_df.PatientID.isin(test_ids)]
        }

    def _create_label_shifted_sets(self, base_df, frac_to_remove):
        """Creates versions of a df with high-density or low-density labels removed."""
        # Shift 1: Remove a fraction of low-density labels (0 and 1)
        low_density_labels = [0, 1]
        df_pos_shift = base_df[~base_df[self.target_label].isin(low_density_labels)].append(
            base_df[base_df[self.target_label].isin(low_density_labels)].sample(frac=1 - frac_to_remove, random_state=33)
        ).sample(frac=1, random_state=33)

        # Shift 2: Remove a fraction of high-density labels (2 and 3)
        high_density_labels = [2, 3]
        df_neg_shift = base_df[~base_df[self.target_label].isin(high_density_labels)].append(
            base_df[base_df[self.target_label].isin(high_density_labels)].sample(frac=1 - frac_to_remove, random_state=33)
        ).sample(frac=1, random_state=33)
        
        return df_pos_shift, df_neg_shift

    def _create_sampler(self, df):
        """Creates a WeightedRandomSampler for the training set."""
        class_counts = df[self.target_label].value_counts()
        if len(class_counts) <= 1: return None
        
        num_samples = len(df)
        class_weights = num_samples / (len(class_counts) * class_counts)
        sample_weights = df[self.target_label].map(class_weights.to_dict()).to_numpy()
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def _create_dataset(self, df, transform_key, counterfactual_scanner: int = None):
        """Helper to create a MammographyDataset instance, handling empty dataframes."""
        if df is None or df.empty: return None
        dataset = MammographyDataset(df, self.image_dir, self.transforms[transform_key], counterfactual_scanner)
        dataset.target_label = self.target_label
        return dataset

    def setup(self, N_SAMPLES=5000, LABELS_TO_REMOVE=0.25):
        """Prepares all datasets for training, validation, and extensive testing."""
        self.target_label = "tissue_density"
        self.split_on = "scanner_model"
        self.group_a_value = 0  # Base scanner

        master_df_raw = prepare_embed_csv(self.csv_file, self.image_dir)
        master_df = master_df_raw[master_df_raw["scanner_model"] != 5].copy() # Exclude one scanner model
        
        splits = self._get_data_splits(master_df)
        train_df, val_df, test_df = splits['train'], splits['val'], splits['test']

        # 1. Create base training and validation sets (only from scanner 0)
        self.train_df_a = train_df[train_df[self.split_on] == self.group_a_value]
        self.val_df_a = val_df[val_df[self.split_on] == self.group_a_value]
        
        # 2. Create base test set and label-shifted versions
        initial_test_df_a = test_df[test_df[self.split_on] == self.group_a_value]
        self.test_df_a = initial_test_df_a.sample(n=min(N_SAMPLES, len(initial_test_df_a)), random_state=33)
        self.test_df_a_remove_pos, self.test_df_a_remove_neg = self._create_label_shifted_sets(self.test_df_a, LABELS_TO_REMOVE)
        
        # 3. Instantiate base and label-shifted datasets
        self.train_dataset = self._create_dataset(self.train_df_a, 'train')
        self.val_dataset = self._create_dataset(self.val_df_a, 'val')
        self.test_a_dataset = self._create_dataset(self.test_df_a, 'val')
        self.test_a_remove_pos_dataset = self._create_dataset(self.test_df_a_remove_pos, 'val')
        self.test_a_remove_neg_dataset = self._create_dataset(self.test_df_a_remove_neg, 'val')

        # 4. Create and instantiate stress test datasets (image corruptions)
        stress_tests = {"gamma": F.adjust_gamma, "contrast": F.adjust_contrast, "brightness": F.adjust_brightness}
        for name, func in stress_tests.items():
            stress_transform = transforms.Compose([transforms.Lambda(lambda img: func(img, 1.7)), *self.transforms['val'].transforms])
            setattr(self, f'test_a_{name}_dataset', self._create_dataset(self.test_df_a, None))
            getattr(self, f'test_a_{name}_dataset').transform = stress_transform

        # 5. Create and instantiate all counterfactual datasets
        cf_scanners = [2, 3, 4]
        for name, df in {'a': self.test_df_a, 'a_remove_pos': self.test_df_a_remove_pos, 'a_remove_neg': self.test_df_a_remove_neg}.items():
            for scanner_id in cf_scanners:
                setattr(self, f'cf_{name}_s{scanner_id}_dataset', self._create_dataset(df, 'val', counterfactual_scanner=scanner_id))
        
        # 6. Create sampler
        self.sampler = self._create_sampler(self.train_df_a)
        print("\nSetup complete.")

    def _get_safe_dataloader(self, dataset, shuffle=False, sampler=None):
        """Returns a DataLoader, or None if the dataset is empty."""
        if not dataset: return None
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=self.num_workers, pin_memory=True)

    # --- Dataloader Access Methods ---
    def train_dataloader(self): return self._get_safe_dataloader(self.train_dataset, sampler=self.sampler)
    def val_dataloader(self): return self._get_safe_dataloader(self.val_dataset, shuffle=False)
    
    # Stress Test Dataloaders
    def test_dataloader_a_gamma(self): return self._get_safe_dataloader(getattr(self, 'test_a_gamma_dataset', None), shuffle=False)
    def test_dataloader_a_contrast(self): return self._get_safe_dataloader(getattr(self, 'test_a_contrast_dataset', None), shuffle=False)
    def test_dataloader_a_brightness(self): return self._get_safe_dataloader(getattr(self, 'test_a_brightness_dataset', None), shuffle=False)

    # Base and Label-Shifted Test Dataloaders
    def test_dataloader_a(self): return self._get_safe_dataloader(self.test_a_dataset, shuffle=False)
    def test_dataloader_a_remove_pos(self): return self._get_safe_dataloader(self.test_a_remove_pos_dataset, shuffle=False)
    def test_dataloader_a_remove_neg(self): return self._get_safe_dataloader(self.test_a_remove_neg_dataset, shuffle=False)
    
    # Counterfactual Dataloaders
    def cf_dataloader_a_s2(self): return self._get_safe_dataloader(getattr(self, 'cf_a_s2_dataset', None), shuffle=False)
    def cf_dataloader_a_s3(self): return self._get_safe_dataloader(getattr(self, 'cf_a_s3_dataset', None), shuffle=False)
    def cf_dataloader_a_s4(self): return self._get_safe_dataloader(getattr(self, 'cf_a_s4_dataset', None), shuffle=False)