"""
INRIA Aerial Image Dataset Loader
"""
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class INRIADataset(Dataset):
    """
    INRIA Aerial Image Dataset for roof segmentation

    Dataset structure:
    - images/: Contains aerial images
    - gt/: Contains ground truth masks
    """

    def __init__(self,
                 images_dir,
                 masks_dir,
                 transform=None,
                 image_size=512,
                 is_training=True):
        """
        Initialize dataset

        Args:
            images_dir: Directory containing aerial images
            masks_dir: Directory containing ground truth masks
            transform: Albumentations transform pipeline
            image_size: Size to resize images and masks
            is_training: Whether this is training set (affects augmentations)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        self.is_training = is_training

        # Get list of image files
        self.image_files = [f for f in os.listdir(images_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

        # Create default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transforms()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Handle different mask naming conventions
        mask_name = self._get_mask_name(img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load image and mask
        image = cv2.imread(img_path)
        if image is None:
            # Handle corrupted images by creating a placeholder
            print(f"Warning: Could not load corrupted image: {img_path}")
            image = np.random.randint(
                100, 200, (self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Ensure image has correct dimensions
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size))

        # Load mask (convert to binary if needed)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Handle corrupted masks
                print(f"Warning: Could not load corrupted mask: {mask_path}")
                mask = np.zeros(
                    (self.image_size, self.image_size), dtype=np.uint8)
            else:
                # Ensure binary mask (0 or 255)
                mask = (mask > 127).astype(np.uint8) * 255
                # Ensure mask has correct dimensions
                if mask.shape != (self.image_size, self.image_size):
                    mask = cv2.resize(mask, (self.image_size, self.image_size))
        else:
            # If no mask found, create empty mask
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            # Double-check dimensions before applying transforms
            assert image.shape[:2] == mask.shape[:
                                                 2], f"Image shape {image.shape[:2]} != mask shape {mask.shape[:2]}"
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert mask to tensor and ensure proper shape
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        # Normalize mask to 0-1
        if mask.max() > 1:
            mask = mask / 255.0

        # Add channel dimension to mask
        mask = mask.unsqueeze(0)

        return {
            'image': image,
            'mask': mask,
            'filename': img_name
        }

    def _get_mask_name(self, img_name):
        """
        Get corresponding mask filename for an image

        Args:
            img_name: Image filename

        Returns:
            Corresponding mask filename
        """
        # Remove extension
        base_name = os.path.splitext(img_name)[0]

        # Try different mask naming conventions
        possible_names = [
            f"{base_name}.png",
            f"{base_name}.tif",
            f"{base_name}.tiff",
            f"{base_name}_mask.png",
            f"{base_name}_gt.png",
            f"{base_name}_label.png"
        ]

        for name in possible_names:
            if os.path.exists(os.path.join(self.masks_dir, name)):
                return name

        # If no mask found, return the first possible name
        return possible_names[0]

    def _get_default_transforms(self):
        """
        Get default transform pipeline based on training mode

        Returns:
            Albumentations transform pipeline
        """
        if self.is_training:
            # Training transforms with augmentations
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.OneOf([
                    A.RandomCrop(self.image_size, self.image_size),
                    A.CenterCrop(self.image_size, self.image_size),
                ], p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation transforms (no augmentation)
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])


def create_train_val_datasets(images_dir, masks_dir, val_split=0.2, image_size=512):
    """
    Create training and validation datasets

    Args:
        images_dir: Directory containing aerial images
        masks_dir: Directory containing ground truth masks
        val_split: Fraction of data to use for validation
        image_size: Size to resize images and masks

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Get list of all image files
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    # Shuffle files
    np.random.shuffle(image_files)

    # Split into train and validation
    split_idx = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Create train dataset
    train_dataset = INRIADataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        is_training=True
    )

    # Create validation dataset
    val_dataset = INRIADataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        is_training=False
    )

    # Override file lists for proper splitting
    train_dataset.image_files = train_files
    val_dataset.image_files = val_files

    return train_dataset, val_dataset


def get_dataloaders(images_dir, masks_dir, batch_size=8, val_split=0.2,
                    image_size=512, num_workers=4):
    """
    Create training and validation dataloaders

    Args:
        images_dir: Directory containing aerial images
        masks_dir: Directory containing ground truth masks
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        image_size: Size to resize images and masks
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        images_dir, masks_dir, val_split, image_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
