import torch
import torchvision.models.segmentation as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F


# Load the trained model for inference
def load_model(model_path, device):
    model = models.deeplabv3_resnet101(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

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
def postprocess_output(output):
    output_predictions = output.argmax(1).squeeze().cpu().numpy()
    return output_predictions


# Overlay mask on image
def overlay_mask_on_image(image, mask, color=(0, 0, 255)):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = color
    overlayed_image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    return overlayed_image


def visualize_softmax_output(output):
    # Get softmax probabilities for the tag class (class 1)
    probs = torch.softmax(output, dim=1)[0, 1, :, :].cpu().numpy()

    plt.imshow(probs, cmap="hot")
    plt.colorbar()
    plt.title("Softmax Output for Tag Class")
    plt.show()


# Visualize segmentation
def visualize_segmentation(image, mask):
    overlayed_image = overlay_mask_on_image(image, mask)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image with Mask Overlay")
    axs[1].axis("off")
    plt.show()


def main(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    image = cv2.imread(image_path)
    input_tensor = preprocess_image(image, device)
    with torch.no_grad():
        output = model(input_tensor)['out']
        visualize_softmax_output(output)
    mask = postprocess_output(output)
    visualize_segmentation(image, mask)


# Example usage
if __name__ == "__main__":
    image_path = "test_img.jpg"  # Replace with your image path
    model_path = "best_deeplabv3_apriltag.pth"  # Replace with your model path
    main(image_path, model_path)
