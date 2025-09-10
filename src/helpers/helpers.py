import os
import torch
import numpy as np
import random
import threading

def set_seed(seed):
    """
    Sets the seed for all relevant random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed} for reproducibility.")

# A necessary helper function to prevent crashes if an image file is missing.
def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Create a lock to safely write to a log file from multiple threads
log_lock = threading.Lock()

def log_broken_image(filepath):
    """Safely appends a broken image path to a log file."""
    with log_lock:
        with open("broken_images.txt", "a") as f:
            f.write(f"{filepath}\n")