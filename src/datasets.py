"""
PyTorch Dataset implementation and image transformation pipelines.

Handles reading image files, applying normalization, and running data augmentation.
Ensures image channel consistency (converting single-channel L and 4-channel RGBA images 
to standard 3-channel RGB).
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def build_transforms(image_size, mean, std, train=True):
    """Construct Torchvision transformation pipeline for training or evaluation."""
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.Normalize(mean=mean, std=std),
        ])

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class CXRDataset(Dataset):
    """Dataset wrapper constructed from manifest DataFrames containing image paths and labels."""

    def __init__(self, manifest_df, transform):
        self.paths = manifest_df["path"].tolist()
        self.labels = manifest_df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        
        # Safely open and convert image to RGB (handles L, RGB, RGBA uniformly)
        with Image.open(path) as raw_img:
            img = raw_img.convert("RGB")
            
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
