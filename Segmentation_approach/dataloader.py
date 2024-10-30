import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class AprilTagDataset(Dataset):
    """
    Custom Dataset class for AprilTag images and masks.
    """
    def __init__(self, images_dir, masks_dir, file_list, transform=None, mask_transform=None):
        """
        Args:
            images_dir (str): Directory with all the images.
            masks_dir (str): Directory with all the masks.
            file_list (list): List of filenames (without extension) to include in the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_list = file_list
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the filename without extension
        filename = self.file_list[idx]

        # Construct full paths
        img_path = os.path.join(self.images_dir, filename + '.jpg')
        mask_path = os.path.join(self.masks_dir, filename + '.png')

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        # Random data augmentation (optional)
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        # Additional augmentations can be added here

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert mask to tensor and ensure it's of type LongTensor
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).long()
        mask[mask > 0] = 1  # Normalize mask to have values 0 and 1

        return image, mask

def get_dataloaders(images_dir, masks_dir, batch_size=16, num_workers=4, train_ratio=0.8):
    """
    Creates DataLoaders for training and validation datasets.
    """
    # Get all filenames without extension
    all_files = [f[:-4] for f in os.listdir(images_dir) if f.endswith('.jpg')]
    all_files.sort()

    # Shuffle the list
    random.seed(42)
    random.shuffle(all_files)

    # Split into training and validation sets
    train_size = int(train_ratio * len(all_files))
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]

    # Define transformations
    image_transforms = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((480, 640), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # Create datasets
    train_dataset = AprilTagDataset(
        images_dir, masks_dir, train_files, transform=image_transforms, mask_transform=mask_transforms
    )

    val_dataset = AprilTagDataset(
        images_dir, masks_dir, val_files, transform=image_transforms, mask_transform=mask_transforms
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
