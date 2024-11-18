# Import necessary libraries
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Function to order the corners consistently
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference of points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left

    return rect

def visualize_masks(image, masks):
    import random
    import matplotlib.patches as patches

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()

    for mask_info in masks:
        mask = mask_info['segmentation']
        color = [random.random(), random.random(), random.random()]
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        for i in range(3):
            img[:, :, i] = color[i]
        ax.imshow(np.dstack((img, mask * 0.35)))

    plt.axis('off')
    plt.show()

def main():
    # Load the image
    image_path = 'tags4.jpg'  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    # Initialize SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to the SAM model checkpoint
    model_type = "vit_h"

    # Load the model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100  # Adjust based on the expected size of the tag
    )

    # Generate masks
    masks = mask_generator.generate(image)

    # Visualize all masks
    visualize_masks(image, masks)

    # Initialize variables to keep track of the best mask
    best_mask = None
    max_confidence = 0
    best_polygon = None

    # Iterate over all generated masks to find the best one
    for mask_info in masks:
        mask = mask_info['segmentation']
        mask_area = mask_info['area']
        mask_score = mask_info['stability_score']

        # Convert the mask to uint8 format
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has 4 sides (quadrilateral)
            if len(approx_polygon) == 4:
                area = cv2.contourArea(approx_polygon)
                x, y, w, h = cv2.boundingRect(approx_polygon)
                aspect_ratio = float(w) / h

                # Additional checks
                if 0.8 < aspect_ratio < 1.2 and area > 500 and mask_score > max_confidence:
                    max_confidence = mask_score
                    best_mask = mask
                    best_polygon = approx_polygon

    if best_mask is None:
        print("Error: Could not find a suitable mask.")
        return

    # Visualize the segmentation mask overlay on the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(best_mask, alpha=0.5)
    plt.title('Best Segmentation Mask Overlay')
    plt.axis('off')
    plt.show()

    # Order the corners of the detected quadrilateral consistently
    corners = best_polygon.reshape(4, 2)
    image_points = order_corners(corners)

    # Display the image with the detected corners marked
    for point in image_points:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)

    cv2.imshow('Detected Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define the real-world coordinates of the tag corners relative to the center
    tag_size = 0.1  # Specify the actual size of your AprilTag in meters
    half_size = tag_size / 2.0

    object_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)

    # Prepare camera parameters (replace with your actual calibration data)
    fx_cam, fy_cam = 800, 800  # Focal lengths in pixels
    cx, cy = image.shape[1] / 2, image.shape[0] / 2  # Principal point coordinates

    camera_matrix = np.array([
        [fx_cam,    0,    cx],
        [   0,   fy_cam, cy],
        [   0,      0,    1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve the Perspective-n-Point problem to find the pose
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Error: Could not solve PnP problem.")
        return

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Output the rotation matrix and translation vector
    print("Rotation Matrix:")
    print(rotation_matrix)
    print("\nTranslation Vector:")
    print(translation_vector)

    # Visualize the pose by drawing coordinate axes on the image
    axis_length = tag_size * 0.5  # Length of the coordinate axes

    # Define the 3D points for the axes
    axis_3D_points = np.float32([
        [0, 0, 0],                   # Origin at the center of the tag
        [axis_length, 0, 0],         # X-axis endpoint
        [0, axis_length, 0],         # Y-axis endpoint
        [0, 0, -axis_length],        # Z-axis endpoint (negative Z in OpenCV)
    ])

    # Project 3D points to the image plane
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
    image_pose = image.copy()
    origin = tuple(image_points_axes[0])  # Origin point (center of the tag)
    cv2.line(image_pose, origin, tuple(image_points_axes[1]), (0, 0, 255), 3)  # X-axis in red
    cv2.line(image_pose, origin, tuple(image_points_axes[2]), (0, 255, 0), 3)  # Y-axis in green
    cv2.line(image_pose, origin, tuple(image_points_axes[3]), (255, 0, 0), 3)  # Z-axis in blue

    # Display the image with pose visualization
    cv2.imshow('Pose Visualization Centered', image_pose)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
