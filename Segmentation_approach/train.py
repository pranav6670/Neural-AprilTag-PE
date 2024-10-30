import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as models
from dataloader import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load DataLoaders
images_dir = 'dataset/images'
masks_dir = 'dataset/masks'
train_loader, val_loader = get_dataloaders(images_dir, masks_dir, batch_size=16, num_workers=4)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model
model = models.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Function to compute IoU
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

# Training loop
num_epochs = 25
train_losses = []
val_losses = []
train_ious = []
val_ious = []

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in train_loader:
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

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_iou = running_iou / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_ious.append(epoch_iou)

    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_running_iou = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks)

            val_running_loss += loss.item() * images.size(0)
            val_running_iou += iou * images.size(0)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_iou = val_running_iou / len(val_loader.dataset)
    val_losses.append(val_epoch_loss)
    val_ious.append(val_epoch_iou)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")

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
