import torch
import torchvision.models.segmentation as models
import cv2
import numpy as np
from torchvision.transforms import functional as F


# Load the trained model for inference
def load_model(model_path, device):
    model = models.deeplabv3_resnet101(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))

    # Load the state_dict and filter out aux_classifier keys
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


# Preprocess image for the model
def preprocess_image(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))  # Resize only if necessary
    image = F.to_tensor(image).to(device)  # Normalize directly with ToTensor
    return image.unsqueeze(0)


# Post-process model output to create binary mask
def postprocess_output(output, threshold=0.5):
    # Apply softmax to get confidence scores for each pixel
    probs = torch.softmax(output, dim=1)
    mask = probs[0, 1, :, :] > threshold  # Binary mask for class 1 (assuming class 1 is the tag)
    return mask.cpu().numpy().astype(np.uint8)


# Overlay mask on image
def overlay_mask_on_image(image, mask, color=(0, 0, 255)):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = color
    overlayed_image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    return overlayed_image


# Live camera inference
def live_camera_inference(model, device):
    cap = cv2.VideoCapture(0)  # Open webcam (0 is usually the default camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame for model input
        input_tensor = preprocess_image(frame, device)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)['out']

        # Post-process output to obtain the mask
        mask = postprocess_output(output, 0.95)

        # Overlay mask on the original frame
        overlayed_frame = overlay_mask_on_image(frame, mask)

        # Display the output
        cv2.imshow('Live Segmentation', overlayed_frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "models/best_deeplabv3_apriltag.pth"  # Replace with your model path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and start live camera inference
    model = load_model(model_path, device)
    live_camera_inference(model, device)
