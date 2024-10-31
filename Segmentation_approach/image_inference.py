import torch
import torchvision.models.segmentation as models
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the trained model for inference
def load_model(model_path, device):
    model = models.deeplabv3_resnet101(weights=None)  # Consistent with training
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Match output channels
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model


# Preprocess image for the model
def preprocess_image(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image / 255.0 - mean) / std  # Normalize
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return image


# Post-process model output to create binary mask
def postprocess_output(output):
    output_predictions = output.argmax(1).squeeze().cpu().numpy()
    return output_predictions


# Visualize original image and segmentation mask side by side
def visualize_segmentation(image, mask):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Segmentation Mask")
    axs[1].axis("off")

    plt.show()


def main(image_path, model_path):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path, device)

    # Load and preprocess image
    image = cv2.imread(image_path)
    input_tensor = preprocess_image(image, device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out']

    # Post-process output to get mask
    mask = postprocess_output(output)

    # Visualize results
    visualize_segmentation(image, mask)


# Example usage
if __name__ == "__main__":
    image_path = "path_to_image.jpg"  # Replace with your image path
    model_path = "best_deeplabv3_apriltag.pth"  # Replace with your model path
    main(image_path, model_path)
