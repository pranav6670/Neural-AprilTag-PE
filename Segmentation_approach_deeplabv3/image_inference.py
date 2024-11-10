import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import matplotlib.pyplot as plt
import torchvision.models.segmentation as models
import torch.nn as nn
import random

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load('models/best_deeplabv3_apriltag.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the transformation for the input image
transform = A.Compose([
    A.Resize(480, 640),  # Resize to the training dimensions
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict_and_overlay(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    original_size = (image.shape[1], image.shape[0])  # Store the original dimensions (width, height)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Resize the mask back to the original image size
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Create an overlay by blending the original image with the mask
    overlay = image_rgb.copy()
    overlay[pred_mask_resized == 1] = [255, 0, 0]  # Color the mask area in red for visibility

    # Blend the overlay with the original image (use alpha blending for transparency)
    alpha = 0.5  # Adjust transparency
    blended_image = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)

    return image_rgb, pred_mask_resized, blended_image

start, end = 0, 2000
# Run inference on the test image
for i in range(10):
    random_number = random.randint(start, end)
    test_image_path = f'../dataset_segmentation_warp/images/image_{random_number}.jpg'
    image_rgb, pred_mask_resized, blended_image = predict_and_overlay(test_image_path)

    # Visualize the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_resized, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(blended_image)
    plt.title("Overlay")
    plt.axis("off")
    plt.show()
