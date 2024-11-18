import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

def detect_tags_with_black_borders(image_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

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

    # Filter masks to find ones with black borders
    selected_mask = None
    for mask in masks:
        # Convert the mask to uint8
        mask_region = (mask['segmentation'] * 255).astype(np.uint8)

        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask_region)

        # Convert to grayscale for border analysis
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Find contours of the mask
        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Get the largest contour (assume it's the relevant tag)
        contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 sides (quadrilateral)
        if len(approx) != 4:
            continue

        # Extract the bounding box for border analysis
        x, y, w, h = cv2.boundingRect(contour)
        border = gray[y:y+h, x:x+w]

        # Check if the border is predominantly black
        border_pixels = border.flatten()
        black_pixel_ratio = np.sum(border_pixels < 50) / len(border_pixels)  # Adjust threshold as needed

        if black_pixel_ratio > 0.6:  # Adjust threshold based on your dataset
            selected_mask = mask
            break

    if selected_mask is None:
        print("No suitable tag with a black border was found.")
        return

    # Visualize the selected mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image[..., ::-1])
    plt.imshow(selected_mask['segmentation'], alpha=0.5, cmap='jet')
    plt.title('Selected Mask with Black Border')
    plt.axis('off')
    plt.show()

    # Extract the selected mask and refine it
    mask = selected_mask['segmentation'].astype(np.uint8) * 255

    # Find contours again for the selected mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found in the mask.")
        return

    # Assume the largest contour is the tag
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 4 sides
    if len(approx_polygon) == 4:
        corners = approx_polygon.reshape(4, 2)
        print("Detected a quadrilateral for the tag.")
    else:
        print("No quadrilateral found in the segmentation.")
        return

    # Visualize the detected corners
    for point in corners:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    cv2.imshow('Detected Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Pose Estimation
    tag_size = 0.1  # Example: 0.1 meters
    half_size = tag_size / 2.0
    object_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)

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
image_path = 'tags4.jpg'
detect_tags_with_black_borders(image_path)
