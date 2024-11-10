import os
import torch
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random

plt.style.use('ggplot')

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# Define the dataset class with advanced data augmentations
class AprilTagDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.image_paths = sorted(
            [img_path for img_path in glob.glob(os.path.join(images_dir, '*.jpg')) if os.path.isfile(img_path)])
        self.mask_paths = sorted(
            [mask_path for mask_path in glob.glob(os.path.join(masks_dir, '*.png')) if os.path.isfile(mask_path)])
        self.transform = transform

    def __len__(self):
        return min(len(self.image_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx], self.mask_paths[idx]

        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing image or mask file: {img_path} or {mask_path}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image, mask = transformed['image'], transformed['mask']

        return image, mask


# Function to get dataloaders
def get_dataloaders(images_dir, masks_dir, batch_size=8, num_workers=4):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(height=480, width=640, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    dataset = AprilTagDataset(images_dir, masks_dir, transform=train_transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# Overlay function
def overlay_prediction(image, mask, pred, alpha=0.5):
    """
    Overlay prediction on the original image with a given alpha for blending.
    Returns the overlay image.
    """
    mask_overlay = np.zeros_like(image)
    mask_overlay[pred == 1] = [255, 0, 0]  # Red overlay for predictions

    overlay = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
    return overlay


# Visualize and save Predictions
def visualize_and_save_predictions(model, val_loader, alpha=0.5, num_images=5, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images_displayed = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                if images_displayed >= num_images:
                    return

                image = images[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)

                mask = masks[i].cpu().numpy()
                pred = preds[i].cpu().numpy()

                # Create overlay
                overlay = overlay_prediction(image, mask, pred, alpha=alpha)

                # Plot the image, ground truth mask, predicted mask, and overlay
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                axs[0].imshow(image)
                axs[0].set_title('Image')
                axs[0].axis('off')

                axs[1].imshow(mask, cmap='gray')
                axs[1].set_title('Ground Truth Mask')
                axs[1].axis('off')

                axs[2].imshow(pred, cmap='gray')
                axs[2].set_title('Predicted Mask')
                axs[2].axis('off')

                axs[3].imshow(overlay)
                axs[3].set_title('Overlay')
                axs[3].axis('off')

                plt.show()

                # Save overlay image
                overlay_path = os.path.join(save_dir, f'overlay_{images_displayed + 1}.png')
                Image.fromarray(overlay).save(overlay_path)
                print(f"Saved overlay image to {overlay_path}")

                images_displayed += 1


if __name__ == "__main__":
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.deeplabv3_resnet101(weights=models.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load('models/best_deeplabv3_apriltag.pth', map_location=device))
    model = model.to(device)

    # Load validation DataLoader
    images_dir = '../dataset_segmentation_warp/images/'
    masks_dir = '../dataset_segmentation_warp/masks/'
    _, val_loader = get_dataloaders(images_dir, masks_dir, batch_size=4, num_workers=4)

    # Visualize and save predictions with overlay
    visualize_and_save_predictions(model, val_loader, alpha=0.5, num_images=5)
