import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt


def detect_inner_tag_with_pose(image_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)

    # Generate masks automatically
    masks = mask_generator.generate(image)

    # Debug: Visualize all generated masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image[..., ::-1])
    for mask in masks:
        plt.contour(mask['segmentation'], colors=['blue'], levels=[0.5])
    plt.title('All Generated Masks')
    plt.axis('off')
    plt.show()

    # Filter masks to find the inner tag mask based on black and white ratios
    selected_mask = None
    best_black_ratio = 0  # To select the most suitable mask
    best_white_ratio = 0

    for idx, mask in enumerate(masks):
        mask_region = (mask['segmentation'] * 255).astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask_region)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Calculate proportions of black and white pixels
        total_pixels = gray.size
        black_pixels = np.sum(gray < 50)  # Black threshold
        white_pixels = np.sum(gray > 200)  # White threshold
        black_ratio = black_pixels / total_pixels
        white_ratio = white_pixels / total_pixels

        print(f"Mask {idx}: Black Ratio = {black_ratio:.2f}, White Ratio = {white_ratio:.2f}")

        # Select mask with the highest black ratio and a reasonable white ratio
        if black_ratio > 0.4 and white_ratio > 0.02:  # Adjust thresholds as needed
            if black_ratio > best_black_ratio or (black_ratio == best_black_ratio and white_ratio > best_white_ratio):
                selected_mask = mask
                best_black_ratio = black_ratio
                best_white_ratio = white_ratio

    if selected_mask is None:
        print("No suitable inner tag mask was found.")
        return

    # Visualize the selected inner mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image[..., ::-1])
    plt.imshow(selected_mask['segmentation'], alpha=0.5, cmap='jet')
    plt.title('Selected Inner Mask')
    plt.axis('off')
    plt.show()

    # Extract the inner mask and find its corners
    mask = selected_mask['segmentation'].astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found for the inner mask.")
        return

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx_polygon) == 4:
        corners = approx_polygon.reshape(4, 2)
        print("Detected a quadrilateral for the inner tag.")
    else:
        print("No quadrilateral detected.")
        return

    # Visualize detected corners
    for point in corners:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    cv2.imshow('Detected Corners on Inner Mask', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Pose Estimation
    tag_size = 0.1  # Example: Tag size in meters
    half_size = tag_size / 2.0
    object_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)

    fx_cam, fy_cam = 800, 800  # Focal lengths (example values)
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
        corners.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Error: Could not solve PnP problem.")
        return

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Output the 6D pose
    print("Rotation Matrix:")
    print(rotation_matrix)
    print("\nTranslation Vector:")
    print(translation_vector)

    # Visualize Pose
    axis_length = tag_size * 0.5
    axis_3D_points = np.float32([
        [0, 0, 0],                   # Origin at center of tag
        [axis_length, 0, 0],         # X-axis
        [0, axis_length, 0],         # Y-axis
        [0, 0, -axis_length],        # Z-axis
    ])

    image_points_axes, _ = cv2.projectPoints(
        axis_3D_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )

    image_points_axes = image_points_axes.reshape(-1, 2).astype(int)
    image_pose = image.copy()
    corner = tuple(image_points_axes[0])
    cv2.line(image_pose, corner, tuple(image_points_axes[1]), (0, 0, 255), 3)  # X-axis
    cv2.line(image_pose, corner, tuple(image_points_axes[2]), (0, 255, 0), 3)  # Y-axis
    cv2.line(image_pose, corner, tuple(image_points_axes[3]), (255, 0, 0), 3)  # Z-axis

    cv2.imshow('Pose Visualization', image_pose)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function
image_path = r"C:\Users\prana\AprilTags\CapturedImages\image_000463_Yaw0.0_Pitch30.0_Roll70.0.png"
detect_inner_tag_with_pose(image_path)
