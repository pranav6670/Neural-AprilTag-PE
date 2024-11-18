import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from pyapriltags import Detector

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

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours again for the selected mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found in the mask.")
        return

    # Assume the largest contour is the tag
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.01 * cv2.arcLength(contour, True)  # Reduced epsilon for higher accuracy
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 4 sides
    if len(approx_polygon) == 4:
        corners = approx_polygon.reshape(4, 2)
        print("Detected a quadrilateral for the tag.")
    else:
        print("No quadrilateral found in the segmentation.")
        return

    # Order the corners consistently
    image_points_sam = order_corners(corners)

    # Visualize the detected corners
    image_sam_corners = image.copy()
    for point in image_points_sam:
        cv2.circle(image_sam_corners, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    cv2.imshow('Detected Corners (SAM)', image_sam_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Pose Estimation using SAM
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

    # Solve PnP using SAM corners
    success_sam, rotation_vector_sam, translation_vector_sam = cv2.solvePnP(
        object_points,
        image_points_sam.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success_sam:
        print("Error: Could not solve PnP problem with SAM method.")
        return

    # Convert rotation vector to rotation matrix
    rotation_matrix_sam, _ = cv2.Rodrigues(rotation_vector_sam)

    # Visualize Pose (SAM method)
    axis_length = tag_size * 0.5
    axis_3D_points = np.float32([
        [0, 0, 0],                   # Origin at center of tag
        [axis_length, 0, 0],         # X-axis
        [0, axis_length, 0],         # Y-axis
        [0, 0, -axis_length],        # Z-axis
    ])

    image_points_axes_sam, _ = cv2.projectPoints(
        axis_3D_points,
        rotation_vector_sam,
        translation_vector_sam,
        camera_matrix,
        dist_coeffs
    )

    image_points_axes_sam = image_points_axes_sam.reshape(-1, 2).astype(int)
    image_pose_sam = image.copy()
    corner_sam = tuple(image_points_axes_sam[0])
    cv2.line(image_pose_sam, corner_sam, tuple(image_points_axes_sam[1]), (0, 0, 255), 3)  # X-axis
    cv2.line(image_pose_sam, corner_sam, tuple(image_points_axes_sam[2]), (0, 255, 0), 3)  # Y-axis
    cv2.line(image_pose_sam, corner_sam, tuple(image_points_axes_sam[3]), (255, 0, 0), 3)  # Z-axis

    # ================================
    # pyAprilTags Method
    # ================================
    # Convert image to grayscale for pyAprilTags
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the AprilTag detector
    detector = Detector(families='tag36h11')

    # Camera parameters for pyAprilTags
    fx, fy = fx_cam, fy_cam
    cx, cy = cx, cy
    camera_params = (fx, fy, cx, cy)

    # Known reference tag size
    tag_size_reference = tag_size  # Use the same tag size as in SAM method

    # Detect tags in the image
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size_reference)

    if not results:
        print("No tags detected by pyAprilTags.")
        return

    # Create a copy of the image for visualization
    image_pose_apriltag = image.copy()

    # Process each detected tag
    for result in results:
        # Extract tag information
        tag_id = result.tag_id
        tag_family = result.tag_family.decode('utf-8')
        tag_center = result.center
        tag_corners = result.corners

        print(f"Detected Tag ID: {tag_id}")
        print(f"Tag Family: {tag_family}")
        print(f"Tag Center: {tag_center}")
        print(f"Tag Corners:\n{tag_corners}")

        # Get pose estimation from pyAprilTags
        t_vector_apriltag = result.pose_t  # Translation vector
        r_matrix_apriltag = result.pose_R  # Rotation matrix

        # Convert rotation matrix to rotation vector
        rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

        # Visualize Pose (pyAprilTags method)
        # Define points for the axes in 3D space
        axis_length = tag_size_reference * 0.5  # Length of the overlayed axis
        axis_3D_points = np.float32([
            [0, 0, 0],                   # Origin at center of tag
            [axis_length, 0, 0],         # X-axis
            [0, axis_length, 0],         # Y-axis
            [0, 0, -axis_length],        # Z-axis
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

        # Draw the coordinate axes on the image (pyAprilTags method)
        corner_apriltag = tuple(image_points_axes_apriltag[0])  # Origin point at center of tag
        cv2.line(image_pose_apriltag, corner_apriltag, tuple(image_points_axes_apriltag[1]), (0, 0, 255), 3)  # X-axis in red
        cv2.line(image_pose_apriltag, corner_apriltag, tuple(image_points_axes_apriltag[2]), (0, 255, 0), 3)  # Y-axis in green
        cv2.line(image_pose_apriltag, corner_apriltag, tuple(image_points_axes_apriltag[3]), (255, 0, 0), 3)  # Z-axis in blue

        # Draw the bounding box around the tag
        for idx in range(4):
            pt1 = tuple(map(int, tag_corners[idx]))
            pt2 = tuple(map(int, tag_corners[(idx + 1) % 4]))
            cv2.line(image_pose_apriltag, pt1, pt2, (0, 255, 0), 2)

        # Draw the center of the tag
        (cX, cY) = (int(tag_center[0]), int(tag_center[1]))
        cv2.circle(image_pose_apriltag, (cX, cY), 5, (0, 0, 255), -1)

        # **Order the corners to find the top-right corner**
        ordered_corners = order_corners(tag_corners)

        # Get the top-right corner
        top_right_corner = ordered_corners[1]
        x, y = int(top_right_corner[0]), int(top_right_corner[1])

        # **Annotate the tag ID at the top-right corner**
        cv2.putText(
            image_pose_apriltag,
            f"ID: {tag_id}",
            (x - 40, y - 10),  # Adjust offsets as needed
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # ================================
    # Comparison of Both Methods
    # ================================
    # Display the images side by side
    combined_image = np.hstack((image_pose_sam, image_pose_apriltag))
    cv2.imshow('Pose Estimation Comparison (Left: SAM, Right: pyAprilTags)', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the rotation matrices and translation vectors
    print("=== SAM Method Pose ===")
    print("Rotation Matrix (SAM):")
    print(rotation_matrix_sam)
    print("Translation Vector (SAM):")
    print(translation_vector_sam.T)

    print("\n=== pyAprilTags Method Pose ===")
    print("Rotation Matrix (pyAprilTags):")
    print(r_matrix_apriltag)
    print("Translation Vector (pyAprilTags):")
    print(t_vector_apriltag.T)

    # Calculate differences between poses
    rotation_diff = rotation_matrix_sam - r_matrix_apriltag
    translation_diff = translation_vector_sam - t_vector_apriltag

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

    # Display the image with the detected tag and ID
    cv2.imshow('Detected Tag with ID', image_pose_apriltag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function
image_path = 'tags4.jpg'  # Replace with the path to your image
detect_tags_with_black_borders(image_path)
