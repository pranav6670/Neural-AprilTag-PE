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

    # Visualize all generated masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image[..., ::-1])
    for mask in masks:
        plt.contour(mask['segmentation'], colors=['blue'], levels=[0.5])
    plt.title('All Generated Masks (SAM)')
    plt.axis('off')
    plt.show()

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
    image_pose_sam = image.copy()

    for result in results:
        tag_id = result.tag_id
        t_vector_apriltag = result.pose_t.flatten()
        r_matrix_apriltag = result.pose_R
        rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

        # Draw cube and pose vectors for PyAprilTags
        draw_cube_on_tags(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix, dist_coeffs, tag_size)
        draw_pose_vectors(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix, dist_coeffs, tag_size)

        # Overlay tag ID and translation vector
        top_right = tuple(map(int, order_corners(result.corners)[1]))
        cv2.putText(image_pose_apriltag, f"ID: {tag_id}", (top_right[0], top_right[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(image_pose_apriltag, f"T: {t_vector_apriltag[0]:.2f}, {t_vector_apriltag[1]:.2f}, {t_vector_apriltag[2]:.2f}",
        #             (top_right[0], top_right[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Process for SAM using PyAprilTags corners
        tag_corners = result.corners
        ordered_corners = order_corners(tag_corners)

        success, rotation_vector_sam, translation_vector_sam = cv2.solvePnP(
            object_points,
            ordered_corners.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            # Draw cube and pose vectors for SAM
            draw_cube_on_tags(image_pose_sam, rotation_vector_sam, translation_vector_sam, camera_matrix, dist_coeffs, tag_size, color=(255, 0, 0))
            draw_pose_vectors(image_pose_sam, rotation_vector_sam, translation_vector_sam, camera_matrix, dist_coeffs, tag_size)

            # Overlay tag ID and translation vector
            cv2.putText(image_pose_sam, f"ID: {tag_id}", (top_right[0], top_right[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image_pose_sam, f"T: {translation_vector_sam[0]:.2f}, {translation_vector_sam[1]:.2f}, {translation_vector_sam[2]:.2f}",
            #             (top_right[0], top_right[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ================================
    # Combined Visualization
    # ================================
    combined_image = np.hstack((image_pose_sam, image_pose_apriltag))

    # Add labels
    label_color = (0, 0, 0)
    font_scale = 1
    thickness = 2
    text_sam = "SAM PE"
    text_pyapriltags = "PyAprilTags PE"
    # combined_image = cv2.putText(combined_image, text_sam, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)
    # combined_image = cv2.putText(combined_image, text_pyapriltags, (combined_image.shape[1] // 2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)

    # Display and save output
    cv2.imshow('Pose Estimation with Cube and Vectors Visualization', combined_image)
    cv2.imwrite('pose_estimation_with_cubes_and_vectors.jpg', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function
image_path = 'tags4.jpg'
detect_tags_with_comparison(image_path)
