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
    sam_poses = {}
    apriltag_poses = {}

    for result in results:
        tag_id = result.tag_id
        t_vector_apriltag = result.pose_t.flatten()
        r_matrix_apriltag = result.pose_R
        rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

        # Print PyAprilTags pose
        print(f"Tag ID: {tag_id} (PyAprilTags)")
        print(f"Translation: {t_vector_apriltag}")
        print(f"Rotation: {rotation_vector_apriltag.flatten()}")

        # Save PyAprilTags poses for comparison
        apriltag_poses[tag_id] = {
            "translation": t_vector_apriltag,
            "rotation": rotation_vector_apriltag.flatten()
        }

        # Overlay tag ID
        tag_corners = result.corners
        ordered_corners = order_corners(tag_corners)
        top_right = tuple(map(int, ordered_corners[1]))
        cv2.putText(
            image_pose_apriltag,
            f"ID: {tag_id}",
            (top_right[0] - 30, top_right[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        # Draw pose axes
        axis_3D_points = np.float32([
            [0, 0, 0], [tag_size * 0.5, 0, 0],
            [0, tag_size * 0.5, 0], [0, 0, -tag_size * 0.5]
        ])
        image_points_axes_apriltag, _ = cv2.projectPoints(
            axis_3D_points, rotation_vector_apriltag, t_vector_apriltag,
            camera_matrix, dist_coeffs
        )
        image_points_axes_apriltag = image_points_axes_apriltag.reshape(-1, 2).astype(int)
        corner = tuple(image_points_axes_apriltag[0])
        cv2.line(image_pose_apriltag, corner, tuple(image_points_axes_apriltag[1]), (0, 0, 255), 3)
        cv2.line(image_pose_apriltag, corner, tuple(image_points_axes_apriltag[2]), (0, 255, 0), 3)
        cv2.line(image_pose_apriltag, corner, tuple(image_points_axes_apriltag[3]), (255, 0, 0), 3)

    # ================================
    # SAM Pose Estimation for PyAprilTags
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
            # Print SAM pose
            print(f"Tag ID: {tag_id} (SAM)")
            print(f"Translation: {translation_vector_sam.flatten()}")
            print(f"Rotation: {rotation_vector_sam.flatten()}")

            # Save SAM poses for comparison
            sam_poses[tag_id] = {
                "translation": translation_vector_sam.flatten(),
                "rotation": rotation_vector_sam.flatten()
            }

            # Overlay tag ID
            top_right = tuple(map(int, ordered_corners[1]))
            cv2.putText(
                image_pose_sam,
                f"ID: {tag_id}",
                (top_right[0] - 30, top_right[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

            # Draw pose axes
            axis_3D_points = np.float32([
                [0, 0, 0], [tag_size * 0.5, 0, 0],
                [0, tag_size * 0.5, 0], [0, 0, -tag_size * 0.5]
            ])
            image_points_axes_sam, _ = cv2.projectPoints(
                axis_3D_points, rotation_vector_sam, translation_vector_sam,
                camera_matrix, dist_coeffs
            )
            image_points_axes_sam = image_points_axes_sam.reshape(-1, 2).astype(int)
            corner = tuple(image_points_axes_sam[0])
            cv2.line(image_pose_sam, corner, tuple(image_points_axes_sam[1]), (0, 0, 255), 3)
            cv2.line(image_pose_sam, corner, tuple(image_points_axes_sam[2]), (0, 255, 0), 3)
            cv2.line(image_pose_sam, corner, tuple(image_points_axes_sam[3]), (255, 0, 0), 3)

    # ================================
    # Compare Norms of Differences
    # ================================
    print("\n=== Norms of Differences ===")
    for tag_id in sam_poses:
        if tag_id in apriltag_poses:
            translation_diff = sam_poses[tag_id]["translation"] - apriltag_poses[tag_id]["translation"]
            rotation_diff = sam_poses[tag_id]["rotation"] - apriltag_poses[tag_id]["rotation"]
            translation_norm = np.linalg.norm(translation_diff)
            rotation_norm = np.linalg.norm(rotation_diff)
            print(f"Tag ID: {tag_id}")
            print(f"Translation Difference Norm: {translation_norm}")
            print(f"Rotation Difference Norm: {rotation_norm}")

    # ================================
    # Comparison of Both Methods
    # ================================
    combined_image = np.hstack((image_pose_sam, image_pose_apriltag))

    # Add labels
    label_color = (0, 0, 0)
    font_scale = 1
    thickness = 2
    text_sam = "SAM Pose Estimation"
    text_pyapriltags = "PyAprilTags Pose Estimation"
    combined_image = cv2.putText(combined_image, text_sam, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)
    combined_image = cv2.putText(combined_image, text_pyapriltags, (combined_image.shape[1] // 2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)

    cv2.imshow('Pose Estimation Comparison', combined_image)
    cv2.imwrite('out.jpg', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function
image_path = 'tags19.png'
detect_tags_with_comparison(image_path)
