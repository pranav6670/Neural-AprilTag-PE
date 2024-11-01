import torch
import torchvision.models.segmentation as models
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(model_path, device):
    """
    Loads the trained DeepLabV3 model with ResNet101 backbone.
    Args:
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model onto.
    Returns:
        model (nn.Module): The loaded model ready for inference.
    """
    # Initialize the model architecture
    model = models.deeplabv3_resnet101(weights=None)
    # Modify the classifier to match the number of classes (background and AprilTag)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_image(image):
    """
    Preprocesses the input image for the model.
    Args:
        image (numpy.ndarray): The input image in RGB format.
    Returns:
        input_tensor (torch.Tensor): The preprocessed image tensor.
    """
    # Define the preprocessing pipeline
    preprocess = A.Compose([
        A.Resize(height=480, width=640),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # Apply preprocessing
    augmented = preprocess(image=image)
    input_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
    return input_tensor

def run_inference(model, input_tensor, device):
    """
    Runs inference on the input tensor.
    Args:
        model (nn.Module): The loaded model.
        input_tensor (torch.Tensor): The preprocessed input tensor.
        device (torch.device): Device to run inference on.
    Returns:
        output (torch.Tensor): The raw output from the model.
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)['out']
    return output

def postprocess_output(output, original_image_shape):
    """
    Postprocesses the model output to generate a segmentation mask.
    Args:
        output (torch.Tensor): The raw output from the model.
        original_image_shape (tuple): The original shape of the input image.
    Returns:
        mask (numpy.ndarray): The segmentation mask as a NumPy array.
    """
    # Get the predicted class for each pixel
    output_predictions = output.argmax(1).squeeze(0).cpu().numpy()
    # Resize the mask to match the original image size
    mask = cv2.resize(output_predictions, (original_image_shape[1], original_image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def visualize_result(image, mask):
    """
    Overlays the segmentation mask on the original image and displays the result.
    Args:
        image (numpy.ndarray): The original image in BGR format.
        mask (numpy.ndarray): The segmentation mask.
    """
    # Create a color map for visualization
    color_map = np.array([[0, 0, 0], [0, 255, 0]])  # Background: black, Foreground (AprilTag): green
    # Map mask to colors
    mask_color = color_map[mask]
    # Overlay mask on the image
    overlay = cv2.addWeighted(image, 0.7, mask_color.astype(np.uint8), 0.3, 0)
    # Display the result
    cv2.imshow('Segmentation Result', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inference_on_image():
    """
    Performs inference on a single test image.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model_path = 'best_deeplabv3_apriltag.pth'  # Adjust the path if necessary
    model = load_model(model_path, device)
    # Load image
    image_path = 'path_to_your_test_image.jpg'  # Replace with your test image path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape
    # Preprocess image
    input_tensor = preprocess_image(image_rgb)
    # Run inference
    output = run_inference(model, input_tensor, device)
    # Postprocess output
    mask = postprocess_output(output, original_shape)
    # Visualize result
    visualize_result(image, mask)

def live_inference():
    """
    Performs inference on a live camera feed.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model_path = 'best_deeplabv3_apriltag.pth'
    model = load_model(model_path, device)
    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape
        # Preprocess frame
        input_tensor = preprocess_image(image_rgb)
        # Run inference
        output = run_inference(model, input_tensor, device)
        # Postprocess output
        mask = postprocess_output(output, original_shape)
        # Visualize result
        color_map = np.array([[0, 0, 0], [0, 255, 0]])  # Adjust colors as needed
        mask_color = color_map[mask]
        overlay = cv2.addWeighted(frame, 0.7, mask_color.astype(np.uint8), 0.3, 0)
        cv2.imshow('Live Segmentation', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # inference_on_image()
    live_inference()
