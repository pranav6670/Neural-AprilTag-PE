import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import torchvision.models.segmentation as models
import torch.nn as nn
import time

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load('models/best_deeplabv3_apriltag.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the transformation for the input image
transform = A.Compose([
    A.Resize(480, 640),  # Adjusted dimensions (Height=480, Width=640)
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict_mask(image):
    # Preprocess the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Resize the mask back to the original image size
    pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred_mask_resized

def overlay_mask(image, mask):
    # Create an overlay by blending the original image with the mask
    overlay = image.copy()
    overlay[mask == 1] = [0, 0, 255]  # Color the mask area in red for visibility

    # Blend the overlay with the original image (use alpha blending for transparency)
    alpha = 0.5  # Adjust transparency
    blended_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended_image

def live_segmentation():
    cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Main loop
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Predict the mask
        pred_mask = predict_mask(frame)

        # Overlay the mask on the original frame
        blended_frame = overlay_mask(frame, pred_mask)

        # Display the resulting frame
        cv2.imshow('Live AprilTag Segmentation', blended_frame)

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}", end='\r')

        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    live_segmentation()
