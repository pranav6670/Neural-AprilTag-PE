import matplotlib.pyplot as plt
import pickle
import torch
from dataloader import get_dataloaders
import torchvision.models.segmentation as models
import numpy as np

# Load training metrics
with open('training_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

train_losses = metrics['train_losses']
val_losses = metrics['val_losses']
train_ious = metrics['train_ious']
val_ious = metrics['val_ious']

num_epochs = len(train_losses)

# Plot Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

# Plot IoU Curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_ious, label='Training IoU')
plt.plot(range(1, num_epochs+1), val_ious, label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('IoU over Epochs')
plt.legend()
plt.savefig('iou_curve.png')
plt.show()

# Function to compute IoU (same as in train_model.py)
def compute_iou(preds, masks):
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

# Visualize Predictions
def visualize_predictions(model, val_loader, num_images=5):
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

if __name__ == "__main__":
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
    model.load_state_dict(torch.load('best_deeplabv3_apriltag.pth', map_location=device))
    model = model.to(device)

    # Load validation DataLoader
    images_dir = '../dataset_segmentation/images'
    masks_dir = '../dataset_segmentation/masks'
    _, val_loader = get_dataloaders(images_dir, masks_dir, batch_size=4, num_workers=4)

    # Visualize predictions
    visualize_predictions(model, val_loader, num_images=5)
