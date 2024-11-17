# Import necessary libraries
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import warnings

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
    # Load the image containing the AprilTag
    image_path = 'tags5.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    original_image = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edged = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold potential AprilTag bounding boxes
    potential_tags = []

    for cnt in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the approximated contour has 4 points
        if len(approx) == 4:
            # Compute the area of the contour
            area = cv2.contourArea(approx)
            if area > 1000:  # Filter out small areas; adjust threshold as needed
                # Check if the contour is convex
                if cv2.isContourConvex(approx):
                    # Add to potential tags
                    potential_tags.append(approx)

    if not potential_tags:
        print("No potential AprilTags found.")
        return

    # Initialize SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # Load the model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # Set the image for the predictor
    predictor.set_image(image)

    # Loop through potential tags and process them
    for idx, tag in enumerate(potential_tags):
        # Get bounding box for the approximated quadrilateral
        x, y, w, h = cv2.boundingRect(tag)
        input_box = np.array([x, y, x + w, y + h])

        # Provide the bounding box to SAM
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Obtain the segmentation mask
        mask = masks[0]

        # Convert the mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        mask_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not mask_contours:
            continue

        # Assume the largest contour corresponds to the tag
        contour = max(mask_contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 sides (quadrilateral)
        if len(approx_polygon) == 4:
            corners = approx_polygon.reshape(4, 2)
        else:
            continue  # Skip if not a quadrilateral

        # Order the corners consistently
        image_points = order_corners(corners)

        # Proceed with pose estimation as before...

        # Define the real-world coordinates of the tag corners relative to the center
        tag_size = 0.1  # For example, 0.1 meters
        half_size = tag_size / 2.0

        object_points = np.array([
            [-half_size, -half_size, 0],
            [ half_size, -half_size, 0],
            [ half_size,  half_size, 0],
            [-half_size,  half_size, 0]
        ], dtype=np.float32)

        # Prepare camera parameters
        # Replace these with your actual camera calibration parameters
        fx_cam, fy_cam = 800, 800  # Focal lengths
        cx, cy = image.shape[1] / 2, image.shape[0] / 2  # Principal point

        camera_matrix = np.array([
            [fx_cam,  0, cx],
            [ 0, fy_cam, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("Error: Could not solve PnP problem for tag index {}.".format(idx))
            continue

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Output the 6D pose
        print("Tag Index: {}".format(idx))
        print("Rotation Matrix:")
        print(rotation_matrix)
        print("\nTranslation Vector:")
        print(translation_vector)

        # Visualize the segmentation mask and pose
        plt.figure(figsize=(10, 10))
        plt.imshow(image[..., ::-1])
        plt.imshow(mask, alpha=0.5)
        plt.title('Segmentation Mask Overlay for Tag Index {}'.format(idx))
        plt.axis('off')
        plt.show()

        # Draw the coordinate axes on the image
        axis_length = tag_size * 0.5
        axis_3D_points = np.float32([
            [0, 0, 0],                   # Origin at center of tag
            [axis_length, 0, 0],         # X-axis
            [0, axis_length, 0],         # Y-axis
            [0, 0, -axis_length],        # Z-axis (negative because OpenCV coordinate system)
        ])

        # Project 3D points to image plane
        image_points_axes, _ = cv2.projectPoints(
            axis_3D_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )

        # Convert points to integer coordinates
        image_points_axes = image_points_axes.reshape(-1, 2).astype(int)

        # Draw the coordinate axes on the image
        image_pose = original_image.copy()
        corner = tuple(image_points_axes[0])  # Origin point at center of tag
        cv2.line(image_pose, corner, tuple(image_points_axes[1]), (0, 0, 255), 3)  # X-axis in red
        cv2.line(image_pose, corner, tuple(image_points_axes[2]), (0, 255, 0), 3)  # Y-axis in green
        cv2.line(image_pose, corner, tuple(image_points_axes[3]), (255, 0, 0), 3)  # Z-axis in blue

        # Show the image with pose visualization
        cv2.imshow('Pose Visualization for Tag Index {}'.format(idx), image_pose)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
