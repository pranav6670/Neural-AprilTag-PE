import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as models
from dataloader import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

def compute_iou(preds, masks):
    """
    Computes the Intersection over Union (IoU) between predicted and ground truth masks.
    Args:
        preds: Predicted masks (batch_size, H, W)
        masks: Ground truth masks (batch_size, H, W)
    Returns:
        Mean IoU over the batch
    """
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    ious = []
    for pred, mask in zip(preds, masks):
        intersection = np.logical_and(pred == 1, mask == 1).sum()
        union = np.logical_or(pred == 1, mask == 1).sum()
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def visualize_predictions(model, val_loader, device, num_images=5):
    """
    Visualizes model predictions on the validation set.
    """
    model.eval()
    images_displayed = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                if images_displayed >= num_images:
                    return

                image = images[i].cpu().numpy().transpose(1, 2, 0)
                # Denormalize if you applied normalization during transforms
                # image = image * std + mean

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

                plt.show()

                images_displayed += 1

def train_model():
    # Load DataLoaders
    images_dir = '../dataset_segmentation/images'
    masks_dir = '../dataset_segmentation/masks'
    train_loader, val_loader = get_dataloaders(images_dir, masks_dir, batch_size=8, num_workers=4)

    # Check device+
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model
    # weights = DeepLabV3_ResNet50_Weights.DEFAULT
    # model = models.deeplabv3_resnet50(weights=weights)
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = models.deeplabv3_mobilenet_v3_large(weights=weights)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = 5
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        # Training Phase
        model.train()
        running_loss = 0.0
        running_iou = 0.0

        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', leave=False)
        for batch_idx, (images, masks) in train_loop:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

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
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)['out']
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)
                iou = compute_iou(preds, masks)

                val_running_loss += loss.item() * images.size(0)
                val_running_iou += iou * images.size(0)

                val_loop.set_postfix(loss=loss.item(), iou=iou)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_iou = val_running_iou / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_ious.append(val_epoch_iou)

        scheduler.step()

        print(f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")

        # Visualize predictions after each epoch (optional)
        # Uncomment the following lines to visualize after each epoch
        # If you prefer to visualize after certain epochs, use an if condition
        # For example: if (epoch + 1) % 5 == 0:

        # Visualize predictions after certain epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print("Visualizing predictions on validation set...")
            visualize_predictions(model, val_loader, device, num_images=5)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'deeplabv3_apriltag.pth')

    # Save training metrics
    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ious': train_ious,
            'val_ious': val_ious
        }, f)

    # Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    # Plot IoU Curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_ious, label='Training IoU')
    plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU over Epochs')
    plt.legend()
    plt.savefig('iou_curve.png')
    plt.show()

if __name__ == '__main__':
    train_model()
