import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


# Function to order the corners consistently
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


# Function to draw a 3D cube on the detected tag
def draw_cube_on_tags(image, rotation_vector, t_vector, camera_matrix, dist_coeffs, tag_size, color=(0, 255, 255)):
    half_size = tag_size / 2
    cube_points = np.float32([
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0],
        [-half_size, half_size, 0],
        [half_size, half_size, -tag_size],
        [half_size, -half_size, -tag_size],
        [-half_size, -half_size, -tag_size],
        [-half_size, half_size, -tag_size]
    ])
    image_points, _ = cv2.projectPoints(cube_points, rotation_vector, t_vector, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2).astype(int)

    # Draw cube edges
    for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0]):
        cv2.line(image, tuple(image_points[i]), tuple(image_points[j]), color, 2)
        cv2.line(image, tuple(image_points[i + 4]), tuple(image_points[j + 4]), color, 2)
        cv2.line(image, tuple(image_points[i]), tuple(image_points[i + 4]), color, 2)


# Function to visualize pose vectors
def draw_pose_vectors(image, rotation_vector, t_vector, camera_matrix, dist_coeffs, tag_size):
    axis_3D_points = np.float32([
        [0, 0, 0],  # Origin
        [tag_size, 0, 0],  # X-axis
        [0, tag_size, 0],  # Y-axis
        [0, 0, -tag_size]  # Z-axis
    ])
    image_points, _ = cv2.projectPoints(axis_3D_points, rotation_vector, t_vector, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2).astype(int)

    origin = tuple(image_points[0])
    # Draw X-axis (red)
    cv2.line(image, origin, tuple(image_points[1]), (0, 0, 255), 2)
    # Draw Y-axis (green)
    cv2.line(image, origin, tuple(image_points[2]), (0, 255, 0), 2)
    # Draw Z-axis (blue)
    cv2.line(image, origin, tuple(image_points[3]), (255, 0, 0), 2)


# Function to convert rotation vector to Euler angles
def rotation_vector_to_euler(rotation_vector):
    """
    Convert a rotation vector to Euler angles (yaw, pitch, roll).
    Yaw, pitch, roll are in degrees.
    """
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)  # Convert to rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)

    singular = sy < 1e-6

    if not singular:  # Standard case
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:  # Gimbal lock case
        yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees([yaw, pitch, roll])  # Convert to degrees


# Function to visualize the tag segmentation with mask
def visualize_tag_segmentation_with_mask(image, mask, corners):
    """
    Visualize the segmentation of the detected tag, its corners, and the mask overlayed on the tag.
    """
    # Create a copy of the original image to overlay the mask
    masked_image = image.copy()

    # Apply the mask with transparency
    alpha = 0.6  # Transparency factor
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    mask_overlay[mask == 1] = (0, 255, 0)  # Green mask

    # Blend the mask overlay with the original image
    cv2.addWeighted(mask_overlay, alpha, masked_image, 1 - alpha, 0, masked_image)

    # Draw the corners on the masked image
    for corner in corners:
        cv2.circle(masked_image, tuple(corner.astype(int)), 5, (0, 0, 255), -1)  # Mark corners in red

    # Visualize the result
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title("Tag Segmentation with Mask and Corners")
    plt.axis("off")
    plt.show()


# Function to visualize the pose comparison
def visualize_pose_comparison(image_pose_sam, image_pose_apriltag):
    """
    Combine and display the SAM and PyAprilTags pose estimations side by side.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_pose_sam, "SAM Pose Estimation", (30, 50), font, 1, (255, 255, 0), 2)
    cv2.putText(image_pose_apriltag, "PyAprilTags Pose Estimation", (30, 50), font, 1, (0, 255, 255), 2)
    combined_image = np.hstack((image_pose_sam, image_pose_apriltag))

    # Display and save the combined result
    cv2.imshow('Pose Estimation Comparison', combined_image)
    cv2.imwrite('pose_estimation_comparison.jpg', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main function
def detect_tags_with_comparison(image_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Initialize camera parameters
    tag_size = 0.1  # Example tag size in meters
    fx_cam, fy_cam = 800, 800
    cx, cy = image.shape[1] / 2, image.shape[0] / 2
    camera_matrix = np.array([[fx_cam, 0, cx], [0, fy_cam, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate masks
    masks = mask_generator.generate(image)

    # Initialize PyAprilTags detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = Detector(families='tag36h11')
    camera_params = (fx_cam, fy_cam, cx, cy)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    # Loop over PyAprilTags results
    for result in results:
        tag_id = result.tag_id

        # PyAprilTags pose
        t_vector_apriltag = result.pose_t.flatten()
        r_matrix_apriltag = result.pose_R
        rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

        # Refine SAM corner detection using PyAprilTags corners
        tag_corners = result.corners
        ordered_corners = order_corners(tag_corners)

        # Find and visualize the best mask
        best_mask = None
        for mask in masks:
            if best_mask is None or np.sum(mask["segmentation"]) > np.sum(best_mask["segmentation"]):
                best_mask = mask

        if best_mask is not None:
            visualize_tag_segmentation_with_mask(image, best_mask["segmentation"], ordered_corners)

        # SAM Pose Estimation
        success, rotation_vector_sam, t_vector_sam = cv2.solvePnP(
            np.array([
                [-tag_size / 2, -tag_size / 2, 0],
                [tag_size / 2, -tag_size / 2, 0],
                [tag_size / 2, tag_size / 2, 0],
                [-tag_size / 2, tag_size / 2, 0]
            ], dtype=np.float32),
            ordered_corners.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # Print pose information
            euler_sam = rotation_vector_to_euler(rotation_vector_sam)
            euler_apriltag = rotation_vector_to_euler(rotation_vector_apriltag)

            print(f"\nTag ID: {tag_id}")
            print("--- PyAprilTags Pose ---")
            print(f"Translation: {t_vector_apriltag}")
            print(f"Euler Angles (Yaw, Pitch, Roll): {euler_apriltag}")

            print("\n--- SAM Pose ---")
            print(f"Translation: {t_vector_sam.flatten()}")
            print(f"Euler Angles (Yaw, Pitch, Roll): {euler_sam}")

            # Compare visualizations
            image_pose_sam = image.copy()
            image_pose_apriltag = image.copy()

            draw_cube_on_tags(image_pose_sam, rotation_vector_sam, t_vector_sam, camera_matrix, dist_coeffs, tag_size, color=(255,255,0))
            draw_pose_vectors(image_pose_sam, rotation_vector_sam, t_vector_sam, camera_matrix, dist_coeffs, tag_size)

            draw_cube_on_tags(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix, dist_coeffs, tag_size, color=(0,255,255))
            draw_pose_vectors(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix, dist_coeffs, tag_size)

            visualize_pose_comparison(image_pose_sam, image_pose_apriltag)


# Path to the input image
image_path = r"C:\Users\prana\AprilTags\CapturedImages\image_000492_Yaw40.0_Pitch30.0_Roll-15.0.png"

# Path to the SAM model checkpoint
sam_checkpoint_path = "sam_vit_h_4b8939.pth"

# Call the function for pose comparison
detect_tags_with_comparison(image_path, sam_checkpoint=sam_checkpoint_path, model_type="vit_h")
