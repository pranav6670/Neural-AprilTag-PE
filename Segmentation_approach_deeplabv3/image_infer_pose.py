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


# Function to order the corners consistently
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference of points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def detect_corners_and_pose(image_rgb, pred_mask_resized, tag_size, camera_matrix, dist_coeffs):
    # Convert the mask to uint8
    mask_uint8 = (pred_mask_resized * 255).astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in the mask.")
        return None, None, None, None

    # Assume the largest contour corresponds to the tag
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 4 sides (quadrilateral)
    if len(approx_polygon) != 4:
        print("Could not find a quadrilateral in the segmentation.")
        return None, None, None, None

    # Reshape and order the corners
    corners = approx_polygon.reshape(4, 2)
    image_points = order_corners(corners)

    # Draw the detected corners on the image
    image_with_corners = image_rgb.copy()
    for point in image_points:
        cv2.circle(image_with_corners, tuple(point.astype(int)), 5, (255, 0, 0), -1)  # Red color in RGB

    # Proceed to pose estimation
    half_size = tag_size / 2.0

    # Define the real-world coordinates of the tag corners
    object_points = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0]
    ], dtype=np.float32)

    # Solve PnP to find the rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Could not solve PnP problem.")
        return None, None, None, None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Return the pose and the image with corners
    return rotation_vector, translation_vector, rotation_matrix, image_with_corners


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


# Specify the path to your test image
test_image_path = '../dataset_segmentation_warp1/images/image_0.jpg'  # Replace with your image path
image_rgb, pred_mask_resized, blended_image = predict_and_overlay(test_image_path)

# Define tag size and camera parameters (replace with your actual calibration data)
tag_size = 0.1  # Tag size in meters
fx_cam, fy_cam = 800, 800  # Focal lengths in pixels
cx, cy = image_rgb.shape[1] / 2, image_rgb.shape[0] / 2  # Principal point coordinates

camera_matrix = np.array([
    [fx_cam, 0, cx],
    [0, fy_cam, cy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Detect corners and estimate pose
rotation_vector, translation_vector, rotation_matrix, image_with_corners = detect_corners_and_pose(
    image_rgb, pred_mask_resized, tag_size, camera_matrix, dist_coeffs)

if rotation_vector is not None and translation_vector is not None:
    print("Rotation Matrix:")
    print(rotation_matrix)
    print("\nTranslation Vector:")
    print(translation_vector)

    # Visualize the pose by drawing coordinate axes on the image
    axis_length = tag_size * 0.5  # Length of the coordinate axes

    # Define the 3D points for the axes
    axis_3D_points = np.float32([
        [0, 0, 0],  # Origin at the center of the tag
        [axis_length, 0, 0],  # X-axis endpoint
        [0, axis_length, 0],  # Y-axis endpoint
        [0, 0, -axis_length],  # Z-axis endpoint (negative Z in OpenCV)
    ])

    # Project 3D points to the image plane using rotation_vector
    image_points_axes, _ = cv2.projectPoints(
        axis_3D_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )

    # Convert projected points to integer coordinates
    image_points_axes = image_points_axes.reshape(-1, 2).astype(int)

    # Draw the coordinate axes on the image
    image_pose = image_rgb.copy()

    # Convert image to BGR for OpenCV drawing functions
    image_pose_bgr = cv2.cvtColor(image_pose, cv2.COLOR_RGB2BGR)
    origin = tuple(image_points_axes[0])  # Origin point (center of the tag)
    cv2.line(image_pose_bgr, origin, tuple(image_points_axes[1]), (0, 0, 255), 3)  # X-axis in red (BGR)
    cv2.line(image_pose_bgr, origin, tuple(image_points_axes[2]), (0, 255, 0), 3)  # Y-axis in green
    cv2.line(image_pose_bgr, origin, tuple(image_points_axes[3]), (255, 0, 0), 3)  # Z-axis in blue

    # Draw pose vectors (optional)
    # Scale the rotation vectors for visualization
    scale = axis_length * 3  # Adjust scale as needed

    # Get the camera center in world coordinates (assuming at origin)
    camera_center_world = np.array([0, 0, 0], dtype=np.float32)

    # Convert rotation vector to rotation matrix
    rotation_matrix_pose, _ = cv2.Rodrigues(rotation_vector)

    # Compute the endpoints of the pose vectors
    x_axis_world = camera_center_world + rotation_matrix_pose[:, 0] * scale
    y_axis_world = camera_center_world + rotation_matrix_pose[:, 1] * scale
    z_axis_world = camera_center_world + rotation_matrix_pose[:, 2] * scale

    # Project the pose vectors onto the image plane
    pose_axes_points = np.vstack((camera_center_world, x_axis_world, y_axis_world, z_axis_world))
    image_points_pose_vectors, _ = cv2.projectPoints(
        pose_axes_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )
    image_points_pose_vectors = image_points_pose_vectors.reshape(-1, 2).astype(int)

    # Draw the pose vectors on the image
    cv2.line(image_pose_bgr, origin, tuple(image_points_pose_vectors[1]), (0, 0, 255), 2)  # X-axis
    cv2.line(image_pose_bgr, origin, tuple(image_points_pose_vectors[2]), (0, 255, 0), 2)  # Y-axis
    cv2.line(image_pose_bgr, origin, tuple(image_points_pose_vectors[3]), (255, 0, 0), 2)  # Z-axis

    # Convert back to RGB for matplotlib
    image_pose = cv2.cvtColor(image_pose_bgr, cv2.COLOR_BGR2RGB)

    # Visualize the results
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(pred_mask_resized, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(image_with_corners)
    plt.title("Detected Corners")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(image_pose)
    plt.title("Pose Visualization")
    plt.axis("off")
    plt.show()
else:
    # If pose estimation failed, show the initial results
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
