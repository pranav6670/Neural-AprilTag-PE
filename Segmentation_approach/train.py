import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import os
import cv2
import glob
import threading
import webbrowser
import time

import warnings
warnings.filterwarnings("ignore")


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
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

def get_dataloaders(images_dir, masks_dir, batch_size=8, num_workers=4):
    # Define transformations
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(height=480, width=640, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Create datasets
    dataset = AprilTagDataset(images_dir, masks_dir, transform=train_transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Update transforms for validation dataset
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

# Define Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        inputs = inputs[:, 1, :, :]
        targets = targets_one_hot[:, 1, :, :]
        intersection = (inputs * targets).sum(dim=(1, 2))
        total = inputs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()

def compute_iou(preds, masks):
    intersection = ((preds == 1) & (masks == 1)).sum(dim=(1, 2)).float()
    union = ((preds == 1) | (masks == 1)).sum(dim=(1, 2)).float()
    iou = intersection / union
    iou[union == 0] = float('nan')
    return torch.nanmean(iou).item()

def launch_tensorboard(logdir):
    import tensorboard
    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', '6006'])
    url = tb.launch()
    print(f"TensorBoard is running at {url}")
    # Open the TensorBoard page in a web browser
    # webbrowser.open(url)

def train_model():
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/experiment1')

    # Start TensorBoard in a separate thread
    tb_thread = threading.Thread(target=launch_tensorboard, args=('runs/',), daemon=True)
    tb_thread.start()
    time.sleep(3)  # Wait a few seconds for TensorBoard to launch

    # Load DataLoaders
    images_dir = '../dataset_segmentation/images'
    masks_dir = '../dataset_segmentation/masks'
    train_loader, val_loader = get_dataloaders(images_dir, masks_dir, batch_size=8, num_workers=8)

    # Fix the validation images for consistent visualization
    fixed_images, fixed_masks = next(iter(val_loader))

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = models.deeplabv3_resnet101(weights=weights)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

    # Freeze backbone layers initially
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Define loss functions and optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Learning rate scheduler
    from torch.optim.lr_scheduler import OneCycleLR
    num_epochs = 100
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=steps_per_epoch, epochs=num_epochs)

    # Early stopping parameters
    early_stopping_patience = 10
    epochs_no_improve = 0
    best_val_iou = 0.0  # For saving the best model

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    accumulation_steps = 1  # Set to higher than 1 if you want to use gradient accumulation

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}")

        # Unfreeze backbone after certain epochs
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True

        # Training Phase
        model.train()
        running_loss = 0.0
        running_iou = 0.0

        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', leave=False)
        optimizer.zero_grad()
        for batch_idx, (images, masks) in train_loop:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)['out']
                loss_ce = criterion_ce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = loss_ce + loss_dice

                loss = loss / accumulation_steps  # Normalize loss if using accumulation

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks)

            running_loss += loss.item() * images.size(0)
            running_iou += iou * images.size(0)

            train_loop.set_postfix(loss=loss.item(), iou=iou)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_ious.append(epoch_iou)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_iou = 0.0

        val_loop = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation', leave=False)
        with torch.no_grad():
            for batch_idx, (images, masks) in val_loop:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)['out']
                    loss_ce = criterion_ce(outputs, masks)
                    loss_dice = criterion_dice(outputs, masks)
                    loss = loss_ce + loss_dice

                preds = torch.argmax(outputs, dim=1)
                iou = compute_iou(preds, masks)

                val_running_loss += loss.item() * images.size(0)
                val_running_iou += iou * images.size(0)

                val_loop.set_postfix(loss=loss.item(), iou=iou)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_iou = val_running_iou / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_ious.append(val_epoch_iou)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_epoch_loss, epoch)
        writer.add_scalar('IoU/train', epoch_iou, epoch)
        writer.add_scalar('IoU/val', val_epoch_iou, epoch)

        print(f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")

        # Early stopping
        # if val_epoch_iou > best_val_iou:
        #         #     best_val_iou = val_epoch_iou
        #         #     epochs_no_improve = 0
        #         #     torch.save(model.state_dict(), 'best_deeplabv3_apriltag.pth')
        #         #     print("Saved Best Model")
        #         # else:
        #         #     epochs_no_improve += 1
        #         #     if epochs_no_improve >= early_stopping_patience:
        #         #         print("Early stopping triggered")
        #         #         break

        # Visualize predictions after certain epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print("Visualizing predictions on validation set...")
            visualize_predictions(model, fixed_images, fixed_masks, device, epoch, num_images=5)

    # Save the last model checkpoint
    torch.save(model.state_dict(), 'last_deeplabv3_apriltag.pth')

    # Save training metrics
    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ious': train_ious,
            'val_ious': val_ious
        }, f)

    # Close TensorBoard writer
    writer.close()

def visualize_predictions(model, fixed_images, fixed_masks, device, epoch, num_images=5, save_dir='visualizations'):
    """
    Visualizes model predictions on the validation set and saves them.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        images = fixed_images.to(device)
        masks = fixed_masks.to(device)
        with autocast():
            outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)

        for i in range(num_images):
            image = images[i].cpu().numpy().transpose(1, 2, 0)
            # Denormalize if you applied normalization during transforms
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            mask = masks[i].cpu().numpy()
            pred = preds[i].cpu().numpy()

            # Plot the image, ground truth mask, and predicted mask
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(image)
            axs[0].set_title('Image')
            axs[0].axis('off')

            axs[1].imshow(mask, cmap='gray')
            axs[1].set_title('Ground Truth Mask')
            axs[1].axis('off')

            axs[2].imshow(pred, cmap='gray')
            axs[2].set_title('Predicted Mask')
            axs[2].axis('off')

            # Save the figure
            save_path = os.path.join(save_dir, f'epoch_{epoch+1}_image_{i}.png')
            plt.savefig(save_path)
            plt.close(fig)

if __name__ == '__main__':
    train_model()
