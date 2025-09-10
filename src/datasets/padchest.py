import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import io
from torchvision import transforms
from torchvision.transforms import functional as F

# --- Helper Functions ---

def collate_fn_skip_none(batch):
    """Collate function that filters out None values from a batch."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def log_broken_image(image_path, log_file="broken_images.log"):
    """Logs the path of a broken image to a file."""
    with open(log_file, "a") as f:
        f.write(f"{image_path}\n")

def prepare_padchest_csv(csv_file):
    """Loads and filters the PadChest CSV file."""
    print("Loading and preparing the master PadChest DataFrame...")
    df = pd.read_csv(csv_file, low_memory=False)

    # List of known invalid or problematic images to exclude
    invalid_filenames = [
        "216840111366964013829543166512013353113303615_02-092-190.png",
        "216840111366964013962490064942014134093945580_01-178-104.png",
    ]
    df = df[~df["ImageID"].isin(invalid_filenames)]

    # Filter for specific projection, non-pediatric, and valid sex
    df = df[df["Projection"] == "PA"].copy()
    df = df[df["Pediatric"] == "No"].copy()
    df = df[df["PatientSex_DICOM"].isin(["M", "F"])].copy()

    def process_labels(row_labels, target_label):
        if isinstance(row_labels, str):
            list_labels = [label.strip() for label in row_labels.strip("[]'").split(',')]
            return target_label in list_labels
        return False

    # Exclude images with 'exclude' or 'suboptimal study' labels
    df['exclude'] = df['Labels'].apply(lambda x: process_labels(x, 'exclude'))
    df['suboptimal study'] = df['Labels'].apply(lambda x: process_labels(x, 'suboptimal study'))
    df = df[~df['exclude'] & ~df['suboptimal study']]

    # Create simplified columns for scanner type, pneumonia label, and sex
    df["ScannerType"] = df["Manufacturer_DICOM"].apply(lambda x: "Phillips" if x == "PhilipsMedicalSystems" else "Imaging")
    df["pneumonia"] = df['Labels'].apply(lambda x: process_labels(x, 'pneumonia')).astype(int)
    df["Sex"] = df["PatientSex_DICOM"]

    return df

# --- PadChest Dataset Class ---

class PadChestDataset(Dataset):
    """Custom PyTorch Dataset for PadChest images."""
    def __init__(self, df, image_dir, transform=None, use_sex_counterfactuals=False, use_scanner_counterfactuals=False):
        self.df = df
        self.image_dir = image_dir
        self.counterfactual_dir = "/vol/biomedic3/bglocker/mscproj/ms4824/stress-testing-framework/src/datasets/counterfactual-images/"
        self.transform = transform
        
        self.image_ids = df["ImageID"].values
        self.labels = df["pneumonia"].values
        self.patient_ids = df["PatientID"].values
        self.sex = df["Sex"].values
        self.scanners = df["ScannerType"].values
        
        self.use_sex_counterfactuals = use_sex_counterfactuals
        self.use_scanner_counterfactuals = use_scanner_counterfactuals

    def __len__(self):
        return len(self.df)

    def _read_image_as_pil(self, idx):
        """Reads an image and returns it as a PIL image, handling counterfactual paths."""
        if self.use_sex_counterfactuals:
            original_sex = "male" if self.sex[idx] == "M" else "female"
            image_path = os.path.join(self.counterfactual_dir, "sex", original_sex, self.image_ids[idx])
        elif self.use_scanner_counterfactuals:
            original_scanner_type = "phillips" if self.scanners[idx] == "Phillips" else "imaging"
            image_path = os.path.join(self.counterfactual_dir, "scanner", original_scanner_type, self.image_ids[idx])
        else:
            image_path = os.path.join(self.image_dir, self.image_ids[idx])

        # Attempt to read with scikit-image, fallback to PIL for truncated images
        try:
            img_array = io.imread(image_path, as_gray=True)
        except Exception:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            pil_img = Image.open(image_path).convert('L')
            img_array = np.array(pil_img)
            ImageFile.LOAD_TRUNCATED_IMAGES = False

        # Normalize to [0, 1] and convert to 8-bit RGB for PIL
        max_val = img_array.max()
        if max_val > 0:
            img_array = img_array / max_val
        
        pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
        return pil_image.convert("RGB")

    def __getitem__(self, idx):
        try:
            image = self._read_image_as_pil(idx)
        except Exception as e:
            image_path = os.path.join(self.image_dir, self.image_ids[idx])
            print(f"ERROR: Could not load image {image_path}. Error: {e}. Skipping.", flush=True)
            log_broken_image(image_path)
            return None
        
        image_tensor = self.transform(image) if self.transform else image
        
        return {
            'x': image_tensor,
            'y': torch.tensor(self.labels[idx], dtype=torch.long),
            'scanner': self.scanners[idx],
            'sex': self.sex[idx],
            'patient_id': self.patient_ids[idx]
        }

# --- PadChest DataModule Class ---

class PadChestDataModule:
    """Manages data loading, splitting, and augmentation for the PadChest dataset."""
    def __init__(self, csv_file, image_dir, batch_size, num_workers=6):
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_col = "pneumonia"
        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def _perform_patient_split(self, master_df):
        """Splits the data into train, validation, and test sets based on patient IDs."""
        print("Performing patient-aware data split...")
        unique_patient_ids = master_df.PatientID.unique()
        train_val_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.20, random_state=33)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=0.10, random_state=33)
        return {
            "train": master_df[master_df.PatientID.isin(train_ids)],
            "val": master_df[master_df.PatientID.isin(val_ids)],
            "test": master_df[master_df.PatientID.isin(test_ids)]
        }

    def _balance_test_set_b(self, df_a, df_b):
        """Downsamples Test Set B to match the label distribution of Test Set A."""
        print("\n--- Adjusting Test Set B to Match Test Set A's Distribution ---")
        dist_a = df_a[self.label_col].value_counts(normalize=True)
        counts_b = df_b[self.label_col].value_counts()

        if len(dist_a) < 2 or len(counts_b) < 2:
            print("Skipping adjustment: A test set has only one class.")
            return df_b

        prop_a_0, prop_a_1 = dist_a.get(0, 0), dist_a.get(1, 0)
        count_b_0, count_b_1 = counts_b.get(0, 0), counts_b.get(1, 0)

        # Determine the limiting class to calculate the new total size
        # Added a check for division by zero
        if prop_a_0 == 0 or prop_a_1 == 0:
            print("Skipping adjustment: Cannot balance due to zero proportion in Test Set A.")
            return df_b
            
        new_total_size = min(count_b_0 / prop_a_0, count_b_1 / prop_a_1)
        new_count_0 = int(new_total_size * prop_a_0)
        new_count_1 = int(new_total_size * prop_a_1)

        df_b_0 = df_b[df_b[self.label_col] == 0].sample(n=new_count_0, random_state=33)
        df_b_1 = df_b[df_b[self.label_col] == 1].sample(n=new_count_1, random_state=33)
        
        df_b_balanced = pd.concat([df_b_0, df_b_1]).sample(frac=1, random_state=33).reset_index(drop=True)
        print(f"Resampled Test Set B size: {len(df_b_balanced)}")
        print("New Test (Group B) Label Distribution:")
        print(df_b_balanced[self.label_col].value_counts(normalize=True))
        return df_b_balanced

    def _create_label_shifted_sets(self, base_df, name, frac_to_remove=0.25):
        """Creates versions of a dataframe with a fraction of positive or negative labels removed."""
        if base_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        print(f"\n--- Creating Label-Shifted Versions of {name} ---")
        pos_samples = base_df[base_df[self.label_col] == 1]
        neg_samples = base_df[base_df[self.label_col] == 0]

        # Create 'remove_pos' version
        pos_kept = pos_samples.sample(frac=1 - frac_to_remove, random_state=33)
        df_remove_pos = pd.concat([pos_kept, neg_samples]).sample(frac=1, random_state=33)
        print(f"Created '{name} Remove Positive'. Size: {len(df_remove_pos)}")

        # Create 'remove_neg' version
        neg_kept = neg_samples.sample(frac=1 - frac_to_remove, random_state=33)
        df_remove_neg = pd.concat([pos_samples, neg_kept]).sample(frac=1, random_state=33)
        print(f"Created '{name} Remove Negative'. Size: {len(df_remove_neg)}")
        
        return df_remove_pos, df_remove_neg
    
    def _create_sampler(self, df):
        """Creates a WeightedRandomSampler to handle class imbalance in the training set."""
        class_counts = df[self.label_col].value_counts()
        if len(class_counts) <= 1:
            return None
        
        num_samples = len(df)
        class_weights = num_samples / (len(class_counts) * class_counts)
        sample_weights = df[self.label_col].map(class_weights.to_dict()).to_numpy()
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def _create_stress_test_sets(self, base_df, name_prefix):
        """Creates datasets with various image corruptions for stress testing."""
        if base_df.empty:
            return
        
        print(f"\n--- Creating Stress Test Datasets for {name_prefix.replace('_', ' ').title()} ---")
        base_transforms = [transforms.Resize((224, 224), antialias=True)]
        tensor_norm_transforms = [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        
        stress_tests = {
            "gamma": transforms.Lambda(lambda img: F.adjust_gamma(img, gamma=1.7)),
            "contrast": transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=1.7)),
            "brightness": transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor=1.5)),
            "blur": transforms.GaussianBlur(kernel_size=(7, 7), sigma=(1.5, 1.5)),
        }

        for name, transform_op in stress_tests.items():
            print(f"🔧 Creating stress test set for: {name.capitalize()}")
            stress_transform = transforms.Compose(base_transforms + [transform_op] + tensor_norm_transforms)
            dataset = PadChestDataset(base_df, self.image_dir, transform=stress_transform)
            setattr(self, f"{name_prefix}_{name}_dataset", dataset)

    def setup(self, split_on="Sex", split_values=("F", "M"), test_limit: int = None):
        """Prepares all data splits, balances test sets, creates label shifts, and instantiates datasets."""
        self.split_on = split_on
        self.group_a_value, self.group_b_value = split_values
        self.use_scanner_counterfactuals = (split_on == "ScannerType")
        self.use_sex_counterfactuals = (split_on == "Sex")

        master_df = prepare_padchest_csv(self.csv_file)
        splits = self._perform_patient_split(master_df)
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

        # 1. Create base splits for Group A and Group B
        self.train_df_a = train_df[train_df[split_on] == self.group_a_value]
        self.val_df_a = val_df[val_df[split_on] == self.group_a_value]
        self.val_df_b = val_df[val_df[split_on] == self.group_b_value]
        self.test_df_a = test_df[test_df[split_on] == self.group_a_value]
        self.test_df_b = test_df[test_df[split_on] == self.group_b_value]

        if test_limit:
            self.test_df_a = self.test_df_a.sample(n=min(test_limit, len(self.test_df_a)), random_state=33)
            self.test_df_b = self.test_df_b.sample(n=min(test_limit, len(self.test_df_b)), random_state=33)

        # 2. Balance Test Set B to match Test Set A's distribution
        self.test_df_b = self._balance_test_set_b(self.test_df_a, self.test_df_b)

        # 3. Create label-shifted versions for both test sets
        self.test_df_a_remove_pos, self.test_df_a_remove_neg = self._create_label_shifted_sets(self.test_df_a, "Test A")
        self.test_df_b_remove_pos, self.test_df_b_remove_neg = self._create_label_shifted_sets(self.test_df_b, "Test B")
        
        # 4. **NEW**: Create image corruption stress tests for Test Set A
        self._create_stress_test_sets(self.test_df_a, 'test_a')

        # 5. Instantiate all datasets
        print("\n--- Instantiating all PyTorch Datasets ---")
        is_cf_scan = self.use_scanner_counterfactuals
        is_cf_sex = self.use_sex_counterfactuals
        
        self.train_dataset = PadChestDataset(self.train_df_a, self.image_dir, self.transforms['train'])
        self.val_dataset = PadChestDataset(self.val_df_a, self.image_dir, self.transforms['val'])
        self.ood_val_dataset = PadChestDataset(self.val_df_b, self.image_dir, self.transforms['val'])
        
        self.test_a_dataset = PadChestDataset(self.test_df_a, self.image_dir, self.transforms['val'])
        self.test_b_dataset = PadChestDataset(self.test_df_b, self.image_dir, self.transforms['val'])
        self.cf_test_a_dataset = PadChestDataset(self.test_df_a, self.image_dir, self.transforms['val'], is_cf_sex, is_cf_scan)            
        self.cf_test_b_dataset = PadChestDataset(self.test_df_b, self.image_dir, self.transforms['val'], is_cf_sex, is_cf_scan)            
        
        self.test_a_remove_pos_dataset = PadChestDataset(self.test_df_a_remove_pos, self.image_dir, self.transforms['val'])
        self.test_a_remove_neg_dataset = PadChestDataset(self.test_df_a_remove_neg, self.image_dir, self.transforms['val'])
        self.cf_test_a_remove_pos_dataset = PadChestDataset(self.test_df_a_remove_pos, self.image_dir, self.transforms['val'], is_cf_sex, is_cf_scan)
        self.cf_test_a_remove_neg_dataset = PadChestDataset(self.test_df_a_remove_neg, self.image_dir, self.transforms['val'], is_cf_sex, is_cf_scan)
        
        self.test_b_remove_pos_dataset = PadChestDataset(self.test_df_b_remove_pos, self.image_dir, self.transforms['val'])
        self.test_b_remove_neg_dataset = PadChestDataset(self.test_df_b_remove_neg, self.image_dir, self.transforms['val'])

        # 6. Create weighted sampler for the training set
        self.sampler = self._create_sampler(self.train_df_a)
        print("\nSetup complete.")

    def _get_safe_dataloader(self, dataset, shuffle=False, sampler=None):
        """Returns a DataLoader, or None if the dataset is empty."""
        if not dataset or len(dataset) == 0: return None
        return DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=shuffle if sampler is None else False,
            sampler=sampler, num_workers=self.num_workers, 
            collate_fn=collate_fn_skip_none, pin_memory=True
        )

    # --- Dataloader Access Methods ---
    def train_dataloader(self): return self._get_safe_dataloader(self.train_dataset, sampler=self.sampler)
    def val_dataloader(self): return self._get_safe_dataloader(self.val_dataset, shuffle=False)
    def ood_val_dataloader(self): return self._get_safe_dataloader(self.ood_val_dataset, shuffle=False)
    
    # Test Set A (in-domain)
    def test_dataloader_a(self): return self._get_safe_dataloader(self.test_a_dataset, shuffle=False)
    def test_dataloader_a_remove_pos(self): return self._get_safe_dataloader(self.test_a_remove_pos_dataset, shuffle=False)
    def test_dataloader_a_remove_neg(self): return self._get_safe_dataloader(self.test_a_remove_neg_dataset, shuffle=False)
    
    # Test Set B (out-of-domain)
    def test_dataloader_b(self): return self._get_safe_dataloader(self.test_b_dataset, shuffle=False)
    def test_dataloader_b_remove_pos(self): return self._get_safe_dataloader(self.test_b_remove_pos_dataset, shuffle=False)
    def test_dataloader_b_remove_neg(self): return self._get_safe_dataloader(self.test_b_remove_neg_dataset, shuffle=False)
    
    # Counterfactuals
    def cf_a_dataloader(self): return self._get_safe_dataloader(self.cf_test_a_dataset, shuffle=False)
    def cf_b_dataloader(self): return self._get_safe_dataloader(self.cf_test_b_dataset, shuffle=False)
    def cf_a_dataloader_remove_pos(self): return self._get_safe_dataloader(self.cf_test_a_remove_pos_dataset, shuffle=False)
    def cf_a_dataloader_remove_neg(self): return self._get_safe_dataloader(self.cf_test_a_remove_neg_dataset, shuffle=False)

    # Stress Test Dataloaders for Test Set A
    def test_dataloader_a_gamma(self): return self._get_safe_dataloader(getattr(self, 'test_a_gamma_dataset', None), shuffle=False)
    def test_dataloader_a_contrast(self): return self._get_safe_dataloader(getattr(self, 'test_a_contrast_dataset', None), shuffle=False)
    def test_dataloader_a_brightness(self): return self._get_safe_dataloader(getattr(self, 'test_a_brightness_dataset', None), shuffle=False)
    def test_dataloader_a_blur(self): return self._get_safe_dataloader(getattr(self, 'test_a_blur_dataset', None), shuffle=False)