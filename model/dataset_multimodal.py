"""
dataset_multimodal.py — Dataset for Multi-Model V6
===================================================
V6 Improvements:
  - Mixup augmentation for better generalization
  - Modality dropout to prevent over-reliance on single modality
  - Slightly stronger augmentation for NS1 samples
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.path_config import resolve_metadata_image_path

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

LABEL_MAP = {"NS1": 0, "NS2": 1, "NS3": 2, "NS4": 3}

ANT_SIZE = 260
RG_SIZE = 224
SL_SIZE = 260

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ═══════════════════════════════════════════════════════════════════
# AUGMENTATION PIPELINES
# ═══════════════════════════════════════════════════════════════════

def get_train_transform_ant_sl(size: int) -> A.Compose:
    """Strong augmentation for anterior segment and slit lamp."""
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=25, p=0.7),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, size // 10),
            hole_width_range=(8, size // 10),
            p=0.3
        ),
        A.Normalize(**NORM),
        ToTensorV2(),
    ])


def get_train_transform_rg() -> A.Compose:
    """Moderate augmentation for red glow (preserve color info but not too conservative)."""
    return A.Compose([
        A.Resize(RG_SIZE, RG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.6),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.Normalize(**NORM),
        ToTensorV2(),
    ])


def get_val_transform(size: int) -> A.Compose:
    """Validation/test transform."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(**NORM),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """Load image using PIL (handles Tamil names)."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        # Do not echo clinical filenames or legacy patient-bearing paths into logs.
        print(f"Warning: Could not load configured image ({type(e).__name__}).")
        return np.zeros((256, 256, 3), dtype=np.uint8)


class MultiModalDatasetV6(Dataset):
    """
    V6 Dataset with mixup support and modality dropout.
    """
    
    def __init__(self, csv_path: str, is_train: bool = True,
                 use_mixup: bool = True, mixup_alpha: float = 0.2,
                 modality_dropout: float = 0.0):
        """
        Args:
            csv_path: Path to CSV
            is_train: Whether to use training augmentations
            use_mixup: Enable mixup augmentation
            mixup_alpha: Beta distribution parameter for mixup
            modality_dropout: Probability of dropping a modality (replacing with zeros)
        """
        df = pd.read_csv(csv_path)
        for column in ("anterior_path", "red_glow_path", "slit_lamp_path"):
            df[column] = df[column].map(
                lambda value: str(resolve_metadata_image_path(value, csv_path))
            )
        mask = df["has_anterior"] & df["has_red_glow"] & df["has_slit_lamp"]
        self.df = df[mask].reset_index(drop=True)
        self.labels = [LABEL_MAP[l] for l in self.df["label"]]
        
        self.is_train = is_train
        self.use_mixup = use_mixup and is_train
        self.mixup_alpha = mixup_alpha
        self.modality_dropout = modality_dropout if is_train else 0.0
        
        if is_train:
            self.transform_ant = get_train_transform_ant_sl(ANT_SIZE)
            self.transform_rg = get_train_transform_rg()
            self.transform_sl = get_train_transform_ant_sl(SL_SIZE)
        else:
            self.transform_ant = get_val_transform(ANT_SIZE)
            self.transform_rg = get_val_transform(RG_SIZE)
            self.transform_sl = get_val_transform(SL_SIZE)
        
        print(f"Dataset V6: {len(self.df)} triplets from {csv_path}")
        counts = pd.Series(self.labels).value_counts().sort_index()
        print("  " + " | ".join(f"NS{i+1}:{counts.get(i, 0)}" for i in range(4)))
        if is_train:
            print(f"  Mixup: {use_mixup} (alpha={mixup_alpha})")
            print(f"  Modality dropout: {modality_dropout}")
    
    def __len__(self):
        return len(self.df)
    
    def _load_and_transform(self, row):
        """Load and transform a single sample."""
        ant = load_image(row["anterior_path"])
        rg = load_image(row["red_glow_path"])
        sl = load_image(row["slit_lamp_path"])
        
        ant = self.transform_ant(image=ant)["image"]
        rg = self.transform_rg(image=rg)["image"]
        sl = self.transform_sl(image=sl)["image"]
        
        return ant, rg, sl
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.labels[idx]
        
        ant, rg, sl = self._load_and_transform(row)
        
        # Modality dropout
        if self.modality_dropout > 0:
            if np.random.random() < self.modality_dropout:
                drop_idx = np.random.randint(3)
                if drop_idx == 0:
                    ant = torch.zeros_like(ant)
                elif drop_idx == 1:
                    rg = torch.zeros_like(rg)
                else:
                    sl = torch.zeros_like(sl)
        
        return ant, rg, sl, label, row["group_id"]
    
    def get_mixup_sample(self, idx):
        """Get a sample for mixup (used by collate function)."""
        row = self.df.iloc[idx]
        label = self.labels[idx]
        ant, rg, sl = self._load_and_transform(row)
        return ant, rg, sl, label


def mixup_collate_fn(batch, dataset, mixup_alpha=0.2, mixup_prob=0.5):
    """
    Collate function with mixup augmentation.
    Primarily mixes NS1 with NS2 to help boundary learning.
    """
    ants, rgs, sls, labels, gids = zip(*batch)
    
    ants = torch.stack(ants)
    rgs = torch.stack(rgs)
    sls = torch.stack(sls)
    labels = torch.tensor(labels)
    
    if np.random.random() < mixup_prob:
        batch_size = len(labels)
        
        # Prefer mixing NS1 with NS2
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        
        # Find indices to mix
        indices = torch.randperm(batch_size)
        
        # Mix
        ants = lam * ants + (1 - lam) * ants[indices]
        rgs = lam * rgs + (1 - lam) * rgs[indices]
        sls = lam * sls + (1 - lam) * sls[indices]
        
        # Soft labels
        labels_onehot = torch.zeros(batch_size, 4)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        labels_mixed = lam * labels_onehot + (1 - lam) * labels_onehot[indices]
        
        return ants, rgs, sls, labels_mixed, gids, True  # True = mixup applied
    
    return ants, rgs, sls, labels, gids, False  # False = no mixup


# ═══════════════════════════════════════════════════════════════════
# WEIGHTED SAMPLER V6
# ═══════════════════════════════════════════════════════════════════

def make_weighted_sampler_v6(dataset, strategy='boundary_focused') -> WeightedRandomSampler:
    """
    V6 sampler with strategies optimized for NS1/NS2 boundary.
    """
    labels = np.array(dataset.labels)
    counts = np.bincount(labels, minlength=4).astype(float)
    
    if strategy == 'boundary_focused':
        # Boost NS2 sampling, reduce NS1
        weights = np.array([0.6, 1.2, 0.9, 1.1])
    elif strategy == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(counts + 1)
        weights = weights / weights.max()
    else:
        # Standard inverse frequency
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.max()
    
    sample_weights = weights[labels]
    
    print(f"Sampler ({strategy}): weights = {[round(w, 3) for w in weights]}")
    
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )


# For backward compatibility
MultiModalDataset = MultiModalDatasetV6
make_weighted_sampler = make_weighted_sampler_v6
