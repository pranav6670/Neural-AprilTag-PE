# Import necessary libraries
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
import torchvision.models.segmentation as models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# Function to order the corners consistently
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left

    return rect

def main():
    # ====================================
    # Set Up Camera Parameters and Tag Size
    # ====================================

    # Camera calibration parameters (replace with your actual values)
    fx_cam, fy_cam = 800, 800  # Focal lengths in pixels
    cx, cy = 320, 240          # Principal point coordinates in pixels

    camera_matrix = np.array([
        [fx_cam,    0,    cx],
        [   0,   fy_cam, cy],
        [   0,      0,    1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Known tag size in meters (replace with your actual tag size)
    tag_size = 0.1  # For example, 0.1 meters
    half_size = tag_size / 2.0

    # ====================================
    # Load and Preprocess the Image
    # ====================================

    # Load the image containing the AprilTag
    image_path = '../dataset_segmentation_warp1/images/image_40.jpg'
    image_color = cv2.imread(image_path)

    if image_color is None:
        print("Error: Could not read the image.")
        return

    # Convert image to grayscale for AprilTag detection
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Convert image to RGB for visualization
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

    # Make copies for drawing
    image_pose_apriltag = image_color.copy()
    image_pose_segmentation = image_color.copy()

    # ====================================
    # Classical Method Using pyAprilTags
    # ====================================

    # Initialize the AprilTag detector
    detector = Detector(families='tag36h11')

    # Detect tags in the image
    results = detector.detect(
        image_gray,
        estimate_tag_pose=True,
        camera_params=(fx_cam, fy_cam, cx, cy),
        tag_size=tag_size
    )

    if not results:
        print("No tags detected by pyAprilTags.")
        return

    # Process the first detected tag
    result = results[0]

    # Get pose estimation from pyAprilTags
    t_vector_apriltag = result.pose_t  # Translation vector
    r_matrix_apriltag = result.pose_R  # Rotation matrix

    # Convert rotation matrix to rotation vector
    rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

    # Draw axes on the image
    axis_length = tag_size * 0.5  # Length of the axes

    # Define points for the axes in 3D space
    axis_3D_points = np.float32([
        [0, 0, 0],                   # Origin at center of tag
        [axis_length, 0, 0],         # X-axis
        [0, axis_length, 0],         # Y-axis
        [0, 0, -axis_length]         # Z-axis (pointing out of the tag)
    ])

    # Project 3D points to image plane
    image_points_axes_apriltag, _ = cv2.projectPoints(
        axis_3D_points,
        rotation_vector_apriltag,
        t_vector_apriltag,
        camera_matrix,
        dist_coeffs
    )

    # Convert points to integer coordinates
    image_points_axes_apriltag = image_points_axes_apriltag.reshape(-1, 2).astype(int)

    # Draw the coordinate axes on the image
    corner_apriltag = tuple(image_points_axes_apriltag[0])  # Origin point at center of tag
    cv2.line(image_pose_apriltag, corner_apriltag, tuple(image_points_axes_apriltag[1]), (0, 0, 255), 3)  # X-axis in red
    cv2.line(image_pose_apriltag, corner_apriltag, tuple(image_points_axes_apriltag[2]), (0, 255, 0), 3)  # Y-axis in green
    cv2.line(image_pose_apriltag, corner_apriltag, tuple(image_points_axes_apriltag[3]), (255, 0, 0), 3)  # Z-axis in blue

    # ====================================
    # Segmentation-Based Method
    # ====================================

    # Load your segmentation model
    # Load your segmentation model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.deeplabv3_resnet101(weights=None)  # Replace with your model
    num_classes = 2  # Background and tag
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    # Updated part to handle aux_classifier keys
    state_dict = torch.load('models/best_deeplabv3_apriltag.pth', map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}
    model.load_state_dict(filtered_state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Define the transformation for the input image
    transform = A.Compose([
        A.Resize(480, 640),  # Resize to the training dimensions
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Preprocess the image
    original_size = (image_rgb.shape[1], image_rgb.shape[0])  # (width, height)
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Resize the mask back to the original image size
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Convert the mask to uint8
    mask_uint8 = (pred_mask_resized * 255).astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in the segmentation mask.")
        return

    # Assume the largest contour corresponds to the tag
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 4 sides (quadrilateral)
    if len(approx_polygon) != 4:
        print("Could not find a quadrilateral in the segmentation.")
        return

    # Reshape and order the corners
    corners = approx_polygon.reshape(4, 2)
    image_points_segmentation = order_corners(corners)

    # Draw the detected corners on the image
    for point in image_points_segmentation:
        cv2.circle(image_pose_segmentation, tuple(point.astype(int)), 5, (0, 0, 255), -1)  # Red color

    # Solve PnP for the segmentation method
    object_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)

    success_segmentation, rotation_vector_segmentation, translation_vector_segmentation = cv2.solvePnP(
        object_points,
        image_points_segmentation.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success_segmentation:
        print("Could not solve PnP for the segmentation method.")
        return

    # Draw axes on the image
    image_points_axes_segmentation, _ = cv2.projectPoints(
        axis_3D_points,
        rotation_vector_segmentation,
        translation_vector_segmentation,
        camera_matrix,
        dist_coeffs
    )

    # Convert points to integer coordinates
    image_points_axes_segmentation = image_points_axes_segmentation.reshape(-1, 2).astype(int)

    # Draw the coordinate axes on the image
    corner_segmentation = tuple(image_points_axes_segmentation[0])  # Origin point at center of tag
    cv2.line(image_pose_segmentation, corner_segmentation, tuple(image_points_axes_segmentation[1]), (0, 0, 255), 3)  # X-axis in red
    cv2.line(image_pose_segmentation, corner_segmentation, tuple(image_points_axes_segmentation[2]), (0, 255, 0), 3)  # Y-axis in green
    cv2.line(image_pose_segmentation, corner_segmentation, tuple(image_points_axes_segmentation[3]), (255, 0, 0), 3)  # Z-axis in blue

    # ====================================
    # Comparison of Both Methods
    # ====================================

    # Display the images side by side
    combined_image = np.hstack((image_pose_apriltag, image_pose_segmentation))
    cv2.imshow('Pose Estimation Comparison (Left: Classical, Right: Segmentation)', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the rotation matrices and translation vectors
    print("=== Classical Method Pose ===")
    print("Rotation Matrix (Classical):")
    print(r_matrix_apriltag)
    print("Translation Vector (Classical):")
    print(t_vector_apriltag.T)

    print("\n=== Segmentation Method Pose ===")
    # Convert rotation vector to rotation matrix
    rotation_matrix_segmentation, _ = cv2.Rodrigues(rotation_vector_segmentation)
    print("Rotation Matrix (Segmentation):")
    print(rotation_matrix_segmentation)
    print("Translation Vector (Segmentation):")
    print(translation_vector_segmentation.T)

    # Calculate differences between poses
    rotation_diff = rotation_matrix_segmentation - r_matrix_apriltag
    translation_diff = translation_vector_segmentation - t_vector_apriltag

    print("\n=== Difference in Poses ===")
    print("Rotation Matrix Difference:")
    print(rotation_diff)
    print("Translation Vector Difference:")
    print(translation_diff.T)

    # Optionally, compute the norm of differences
    rotation_diff_norm = np.linalg.norm(rotation_diff)
    translation_diff_norm = np.linalg.norm(translation_diff)

    print("\nNorm of Rotation Matrix Difference:", rotation_diff_norm)
    print("Norm of Translation Vector Difference:", translation_diff_norm)

if __name__ == "__main__":
    main()
