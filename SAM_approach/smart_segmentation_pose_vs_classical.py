import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from pyapriltags import Detector


# Function to order the corners consistently
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left
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
    # Project 3D points to 2D
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


# Function to convert rotation vectors to Euler angles
def rotation_vector_to_euler(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])


def detect_tags_with_comparison(image_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate masks automatically
    masks = mask_generator.generate(image)

    # Initialize camera parameters
    tag_size = 0.1  # Example tag size in meters
    half_size = tag_size / 2.0
    object_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)
    fx_cam, fy_cam = 800, 800
    cx, cy = image.shape[1] / 2, image.shape[0] / 2
    camera_matrix = np.array([[fx_cam, 0, cx], [0, fy_cam, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # ================================
    # PyAprilTags Method
    # ================================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = Detector(families='tag36h11')
    camera_params = (fx_cam, fy_cam, cx, cy)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    image_pose_apriltag = image.copy()
    sam_poses = {}
    apriltag_poses = {}

    for result in results:
        tag_id = result.tag_id
        t_vector_apriltag = result.pose_t.flatten()
        r_matrix_apriltag = result.pose_R
        rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

        apriltag_poses[tag_id] = {
            "translation": t_vector_apriltag,
            "rotation": rotation_vector_apriltag.flatten()
        }

        draw_cube_on_tags(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix, dist_coeffs, tag_size, color=(0, 255, 255))
        draw_pose_vectors(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix, dist_coeffs, tag_size)

    # ================================
    # SAM Pose Estimation
    # ================================
    image_pose_sam = image.copy()
    for result in results:
        tag_corners = result.corners
        tag_id = result.tag_id
        ordered_corners = order_corners(tag_corners)

        success, rotation_vector_sam, translation_vector_sam = cv2.solvePnP(
            object_points,
            ordered_corners.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            sam_poses[tag_id] = {
                "translation": translation_vector_sam.flatten(),
                "rotation": rotation_vector_sam.flatten()
            }

            draw_cube_on_tags(image_pose_sam, rotation_vector_sam, translation_vector_sam, camera_matrix, dist_coeffs, tag_size, color=(255, 255, 0))
            draw_pose_vectors(image_pose_sam, rotation_vector_sam, translation_vector_sam, camera_matrix, dist_coeffs, tag_size)

    # ================================
    # Detailed Pose Values and Differences
    # ================================
    for tag_id in sam_poses:
        if tag_id in apriltag_poses:
            # Get translation vectors
            translation_sam = sam_poses[tag_id]["translation"]
            translation_apriltag = apriltag_poses[tag_id]["translation"]
            translation_diff = translation_sam - translation_apriltag
            translation_norm = np.linalg.norm(translation_diff)

            # Get rotation vectors and convert to Euler angles
            rotation_sam = sam_poses[tag_id]["rotation"]
            rotation_apriltag = apriltag_poses[tag_id]["rotation"]
            rotation_diff_vector = rotation_sam - rotation_apriltag
            rotation_diff_euler = rotation_vector_to_euler(rotation_sam) - rotation_vector_to_euler(rotation_apriltag)
            rotation_diff_norm = np.linalg.norm(rotation_diff_euler)

            # Print detailed information
            print(f"\nTag ID: {tag_id}")
            print(f"--- SAM Pose ---")
            print(f"Translation: {translation_sam}")
            print(f"Rotation (Vector): {rotation_sam}")
            print(f"Rotation (Euler): {rotation_vector_to_euler(rotation_sam)}")

            print(f"--- PyAprilTags Pose ---")
            print(f"Translation: {translation_apriltag}")
            print(f"Rotation (Vector): {rotation_apriltag}")
            print(f"Rotation (Euler): {rotation_vector_to_euler(rotation_apriltag)}")

            print(f"--- Differences ---")
            print(f"Translation Difference: {translation_diff}")
            print(f"Translation Difference Norm: {translation_norm}")
            print(f"Rotation Difference (Vector): {rotation_diff_vector}")
            print(f"Rotation Difference (Euler): {rotation_diff_euler}")
            print(f"Euler Rotation Difference Norm: {rotation_diff_norm}")

            # ================================
            # Visualization Comparison
            # ================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_pose_sam, "SAM Pose Estimation", (30, 50), font, 1, (255, 255, 0), 2)
        cv2.putText(image_pose_apriltag, "PyAprilTags Pose Estimation", (30, 50), font, 1, (0, 255, 255), 2)
        combined_image = np.hstack((image_pose_sam, image_pose_apriltag))

        # Show and save comparison image
        cv2.imshow('Pose Estimation Comparison', combined_image)
        cv2.imwrite('pose_estimation_comparison.jpg', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Call the function
image_path = r"C:\Users\prana\AprilTags\CapturedImages\image_000500_Yaw20.0_Pitch-10.0_Roll90.0.png"
detect_tags_with_comparison(image_path)
