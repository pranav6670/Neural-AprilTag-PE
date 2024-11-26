import os
import cv2
import numpy as np
import torch
import pickle
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


# Function to calculate pose error
def calculate_pose_error(t_sam, t_apriltag, r_sam, r_apriltag):
    """Calculate translation and rotation differences (norms)."""
    translation_error = np.linalg.norm(t_sam - t_apriltag)

    rotation1 = R.from_rotvec(r_sam.flatten())
    rotation2 = R.from_rotvec(r_apriltag.flatten())
    rotation_diff = rotation1.inv() * rotation2
    rotation_error = rotation_diff.magnitude()

    return translation_error, rotation_error


# Main function to process a single image
def process_image(image_path, sam, mask_generator, detector, camera_matrix, dist_coeffs, tag_size):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image {image_path}.")
        return None

    # Generate masks
    masks = mask_generator.generate(image)

    # Initialize PyAprilTags detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=(camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]), tag_size=tag_size)

    if not results:
        print(f"No tags detected in {image_path}.")
        return None

    # Process first detected tag (assuming one tag per image)
    result = results[0]

    # PyAprilTags pose
    t_vector_apriltag = result.pose_t.flatten()
    r_matrix_apriltag = result.pose_R
    rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

    # Refine SAM corner detection using PyAprilTags corners
    tag_corners = result.corners
    ordered_corners = order_corners(tag_corners)

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

    if not success:
        print(f"SAM Pose Estimation failed for {image_path}.")
        return None

    # Calculate pose error
    translation_error, rotation_error = calculate_pose_error(
        t_vector_sam.flatten(), t_vector_apriltag, rotation_vector_sam.flatten(), rotation_vector_apriltag
    )

    # Compute Euler angles
    euler_sam = rotation_vector_to_euler(rotation_vector_sam)
    euler_apriltag = rotation_vector_to_euler(rotation_vector_apriltag)

    return {
        "image_path": image_path,
        "t_sam": t_vector_sam.flatten(),
        "t_apriltag": t_vector_apriltag,
        "r_sam": rotation_vector_sam.flatten(),
        "r_apriltag": rotation_vector_apriltag.flatten(),
        "euler_sam": euler_sam,
        "euler_apriltag": euler_apriltag,
        "translation_error": translation_error,
        "rotation_error": rotation_error
    }


# Main function to process all images in a directory
def process_directory(directory_path, output_pickle_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Initialize PyAprilTags detector
    detector = Detector(families='tag36h11')

    # Camera parameters
    tag_size = 0.1  # Example tag size in meters
    fx_cam, fy_cam = 800, 800
    cx, cy = 960, 540  # Assuming a 1920x1080 image resolution
    camera_matrix = np.array([[fx_cam, 0, cx], [0, fy_cam, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Process each image
    results = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing {image_path}...")
            result = process_image(image_path, sam, mask_generator, detector, camera_matrix, dist_coeffs, tag_size)
            if result:
                results.append(result)

    # Save results to a pickle file
    with open(output_pickle_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {output_pickle_path}.")

    return results


# Function to visualize results
def visualize_results(results):
    # Extract data
    translation_errors = [r["translation_error"] for r in results]
    rotation_errors = [r["rotation_error"] for r in results]
    image_names = [os.path.basename(r["image_path"]) for r in results]

    # Plot translation errors
    plt.figure(figsize=(12, 6))
    plt.bar(image_names, translation_errors, color="blue", alpha=0.7, label="Translation Error")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Translation Error (m)")
    plt.title("Translation Errors for Images")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot rotation errors
    plt.figure(figsize=(12, 6))
    plt.bar(image_names, rotation_errors, color="red", alpha=0.7, label="Rotation Error")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rotation Error (radians)")
    plt.title("Rotation Errors for Images")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    directory_path = r"C:\Users\prana\AprilTags\CapturedImages"
    output_pickle_path = r"pose_analysis_results.pkl"
    sam_checkpoint_path = "sam_vit_h_4b8939.pth"

    # Process directory and save results
    results = process_directory(directory_path, output_pickle_path, sam_checkpoint=sam_checkpoint_path)

    # Visualize results
    visualize_results(results)
