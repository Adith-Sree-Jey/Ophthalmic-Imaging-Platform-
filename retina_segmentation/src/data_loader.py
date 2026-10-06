import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


VALID_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _sorted_imgs(folder):
    paths = [p for p in glob.glob(os.path.join(folder, "*")) if os.path.splitext(p)[1].lower() in VALID_IMG_EXTS]
    return sorted(paths)


class RetinaDataset(Dataset):
    """
    Retina vessel segmentation dataset with Albumentations transforms.

    Assumptions:
    - Images and masks share the same filenames (e.g., img_001.png ↔ img_001.png)
    - Masks are single-channel, binary-like (0/255). We binarize to {0,1}.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        augment: bool = False,
        img_size: int = 512,
        random_crop: int | None = None,
        add_green: bool = False,  # if True, append green channel => 4-channel input
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_paths = _sorted_imgs(image_dir)
        self.mask_paths = _sorted_imgs(mask_dir)
        self.augment = augment
        self.img_size = img_size
        self.random_crop = random_crop
        self.add_green = add_green

        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")
        if len(self.mask_paths) == 0:
            raise FileNotFoundError(f"No masks found in {mask_dir}")
        if len(self.img_paths) != len(self.mask_paths):
            # We'll try to match by filename
            img_names = {os.path.basename(p) for p in self.img_paths}
            mask_names = {os.path.basename(p) for p in self.mask_paths}
            shared = sorted(img_names & mask_names)
            if not shared:
                raise RuntimeError("Image/mask filenames do not match. Ensure identical names.")
            self.img_paths = [os.path.join(image_dir, n) for n in shared]
            self.mask_paths = [os.path.join(mask_dir, n) for n in shared]

        # --- Transforms ---
        # Note: For masks, Albumentations uses nearest neighbor automatically.
        aug_list = [
            A.Resize(self.img_size, self.img_size),
        ]
        if self.augment:
            aug_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
                A.ElasticTransform(p=0.3, alpha=1, sigma=50, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomBrightnessContrast(p=0.3),
                A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.25),
            ])
        if self.random_crop:
            # Apply AFTER resize so crops are consistent
            aug_list.append(A.RandomCrop(self.random_crop, self.random_crop, p=1.0))

        aug_list.extend([
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ])
        self.transform = A.Compose(aug_list)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # --- Read image as RGB ---
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Read mask as grayscale ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = (mask > 127).astype("float32")  # binarize to {0,1}

        # --- Albumentations (applies same spatial ops to image/mask) ---
        data = self.transform(image=image, mask=mask)
        image_t = data["image"]           # torch.FloatTensor [3, H, W], normalized
        mask_t = data["mask"].unsqueeze(0)  # torch.FloatTensor [1, H, W], 0/1

        if self.add_green:
            # Append green channel as an extra feature (shape becomes [4, H, W])
            green = image_t[1:2, ...]  # channel index 1 is green in RGB
            image_t = torch.cat([image_t, green], dim=0)

        return image_t, mask_t


def get_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    test_img_dir: str,
    test_mask_dir: str,
    batch_size: int = 2,
    img_size: int = 512,
    random_crop: int | None = None,
    add_green: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Returns:
      train_loader: augmented dataset
      test_loader:  eval dataset (no aug except resize/normalize)
    """
    train_ds = RetinaDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        augment=True,
        img_size=img_size,
        random_crop=random_crop,
        add_green=add_green,
    )
    test_ds = RetinaDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        augment=False,
        img_size=img_size,
        random_crop=None,
        add_green=add_green,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


def get_train_val_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    batch_size: int = 2,
    img_size: int = 512,
    random_crop: int | None = None,
    add_green: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Create deterministic train/validation loaders from ``Data/train`` only.

    Separate dataset instances are used so augmentation is enabled for the
    training subset and disabled for validation. ``Data/test`` is deliberately
    not accepted by this function and remains untouched for final evaluation.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    train_dataset = RetinaDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        augment=True,
        img_size=img_size,
        random_crop=random_crop,
        add_green=add_green,
    )
    validation_dataset = RetinaDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        augment=False,
        img_size=img_size,
        random_crop=None,
        add_green=add_green,
    )
    sample_count = len(train_dataset)
    if sample_count < 2:
        raise ValueError("At least two training samples are required for a held-out validation split.")

    validation_count = max(1, min(sample_count - 1, round(sample_count * val_fraction)))
    generator = torch.Generator().manual_seed(split_seed)
    shuffled_indices = torch.randperm(sample_count, generator=generator).tolist()
    validation_indices = shuffled_indices[:validation_count]
    training_indices = shuffled_indices[validation_count:]

    train_loader = DataLoader(
        Subset(train_dataset, training_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, validation_loader


if __name__ == "__main__":
    # Quick sanity check
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_img = os.path.join(BASE_DIR, "Data", "train", "image")
    train_msk = os.path.join(BASE_DIR, "Data", "train", "mask")
    test_img = os.path.join(BASE_DIR, "Data", "test", "image")
    test_msk = os.path.join(BASE_DIR, "Data", "test", "mask")

    tl, vl = get_dataloaders(
        train_img, train_msk,
        test_img, test_msk,
        batch_size=2, img_size=512,
        random_crop=None,        # set e.g. 384 to enable patch training
        add_green=False,         # set True to append green channel (requires model.in_channels=4)
        num_workers=0
    )

    x, y = next(iter(tl))
    print("Train batch image shape:", x.shape)  # [B, 3 or 4, H, W]
    print("Train batch mask  shape:", y.shape)  # [B, 1, H, W]
