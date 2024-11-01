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
import warnings
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# Configuration dictionary
config = {
    'seed': 42,
    'batch_size': 8,
    'num_workers': 16,
    'num_epochs': 25,
    'learning_rate': 0.0001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'save_interval': 5,  # Save model every 5 epochs
    'images_dir': '../dataset_segmentation/images',
    'masks_dir': '../dataset_segmentation/masks',
    'visualizations_dir': 'visualizations',
    'model_save_path': 'checkpoints',
    'log_dir': 'runs',
}


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(config['seed'])

# Launch TensorBoard using the TensorBoard API
def launch_tensorboard(logdir):
    import tensorboard
    from tensorboard import program
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', '6006'])
    url = tb.launch()
    print(f"TensorBoard is running at {url}")

# Dataset class with augmentations
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
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    dataset = AprilTagDataset(images_dir, masks_dir, transform=train_transform)
    val_size = int(0.3 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    return train_loader, val_loader


# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce_loss(inputs, targets)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        focal_loss = self.alpha * focal_loss
        return focal_loss.mean()


# Compute metrics
def compute_metrics(preds, masks):
    preds_flat = preds.view(-1).cpu().numpy()
    masks_flat = masks.view(-1).cpu().numpy()
    precision = precision_score(masks_flat, preds_flat, zero_division=0)
    recall = recall_score(masks_flat, preds_flat, zero_division=0)
    f1 = f1_score(masks_flat, preds_flat, zero_division=0)
    return precision, recall, f1


def compute_iou(preds, masks):
    intersection = ((preds == 1) & (masks == 1)).sum(dim=(1, 2)).float()
    union = ((preds == 1) | (masks == 1)).sum(dim=(1, 2)).float()
    iou = torch.zeros_like(intersection)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou.mean().item()


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, scaler, writer):
    model.train()
    running_loss, running_iou, running_precision, running_recall, running_f1 = 0, 0, 0, 0, 0
    for images, masks in tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}]', leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)['out']
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(outputs, dim=1)
        iou = compute_iou(preds, masks)
        precision, recall, f1 = compute_metrics(preds, masks)

        # Accumulate metrics
        running_loss += loss.item()
        running_iou += iou
        running_precision += precision
        running_recall += recall
        running_f1 += f1

    return running_loss / len(train_loader), running_iou / len(train_loader), \
           running_precision / len(train_loader), running_recall / len(train_loader), running_f1 / len(train_loader)


def validate_one_epoch(epoch, model, val_loader, criterion, device, writer):
    model.eval()
    running_loss, running_iou, running_precision, running_recall, running_f1 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f'Validation Epoch [{epoch + 1}]', leave=False):
            images, masks = images.to(device), masks.to(device)
            with autocast():
                outputs = model(images)['out']
                loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks)
            precision, recall, f1 = compute_metrics(preds, masks)

            running_loss += loss.item()
            running_iou += iou
            running_precision += precision
            running_recall += recall
            running_f1 += f1

    return running_loss / len(val_loader), running_iou / len(val_loader), \
           running_precision / len(val_loader), running_recall / len(val_loader), running_f1 / len(val_loader)


def train_model(config):
    os.makedirs(config['model_save_path'], exist_ok=True)
    os.makedirs(config['visualizations_dir'], exist_ok=True)

    launch_tensorboard(config['log_dir'])

    writer = SummaryWriter(log_dir=config['log_dir'])
    train_loader, val_loader = get_dataloaders(config['images_dir'], config['masks_dir'],
                                               batch_size=config['batch_size'], num_workers=config['num_workers'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = FocalLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()
    early_stopping_patience = config['early_stopping_patience']
    epochs_no_improve = 0
    best_val_f1 = 0.0

    for epoch in range(config['num_epochs']):
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, model.backbone.parameters())})

        train_loss, train_iou, train_prec, train_recall, train_f1 = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device, scaler, writer)
        val_loss, val_iou, val_prec, val_recall, val_f1 = validate_one_epoch(epoch, model, val_loader, criterion, device, writer)

        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(config['model_save_path'], 'best_model.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping due to no improvement in F1 score.")
                break

        scheduler.step(val_loss)
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
        writer.add_scalars('Precision', {'train': train_prec, 'val': val_prec}, epoch)
        writer.add_scalars('Recall', {'train': train_recall, 'val': val_recall}, epoch)
        writer.add_scalars('F1-Score', {'train': train_f1, 'val': val_f1}, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)


if __name__ == '__main__':
    train_model(config)
