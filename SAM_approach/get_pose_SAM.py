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

# Global variables for mouse callback
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1
fx, fy = -1, -1
mode = 'box'  # 'box' or 'point'
points = []

def select_prompt(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, mode, image_display, points

    if mode == 'box':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                fx, fy = x, y
                img_copy = image_display.copy()
                cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 2)
                cv2.imshow('Select Prompt', img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            cv2.rectangle(image_display, (ix, iy), (fx, fy), (0, 255, 0), 2)
            cv2.imshow('Select Prompt', image_display)

    elif mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select Prompt', image_display)
            points.append((x, y))

def main():
    global mode, image_display, points, fx, fy, drawing

    # Load the image containing the AprilTag
    image_path = '../dataset_segmentation_warp/images/image_75.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    image_display = image.copy()

    # Prompt the user to select the mode
    print("Select prompt mode:")
    print("1. Bounding Box")
    print("2. Point(s)")
    mode_input = input("Enter 1 or 2: ")

    if mode_input == '1':
        mode = 'box'
    elif mode_input == '2':
        mode = 'point'
        points = []
    else:
        print("Invalid input.")
        return

    # Set up the window and mouse callback
    cv2.namedWindow('Select Prompt')
    cv2.setMouseCallback('Select Prompt', select_prompt)

    print("Instructions:")
    if mode == 'box':
        print("- Draw a bounding box around the AprilTag by clicking and dragging.")
    elif mode == 'point':
        print("- Click on the image to select points inside the AprilTag.")
        print("- Press 'q' when done selecting points.")

    cv2.imshow('Select Prompt', image_display)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if mode == 'box' and not drawing and (fx != -1 and fy != -1):
            break
        if mode == 'point' and key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Initialize SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # Load the model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # Set the image for the predictor
    predictor.set_image(image)

    # Provide prompts to SAM
    if mode == 'box':
        input_box = np.array([ix, iy, fx, fy])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
    elif mode == 'point':
        if len(points) == 0:
            print("No points selected.")
            return
        input_point = np.array(points)
        input_label = np.ones(len(points))  # Labels: 1 for foreground
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
    else:
        print("Invalid mode.")
        return

    # Obtain the segmentation mask
    mask = masks[0]

    # Visualize the segmentation mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image[..., ::-1])
    plt.imshow(mask, alpha=0.5)
    plt.title('Segmentation Mask Overlay')
    plt.axis('off')
    plt.show()

    # Convert the mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found in the mask.")
        return

    # Assume the largest contour corresponds to the tag
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Debug: Draw the contour approximation
    contour_image = image.copy()
    cv2.drawContours(contour_image, [approx_polygon], -1, (0, 255, 0), 2)
    cv2.imshow('Contour Approximation', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Check if the polygon has 4 sides (quadrilateral)
    if len(approx_polygon) == 4:
        corners = approx_polygon.reshape(4, 2)
    else:
        # Try to find a quadrilateral among all contours
        found_quad = False
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_polygon) == 4:
                corners = approx_polygon.reshape(4, 2)
                found_quad = True
                break
        if not found_quad:
            print("Error: Could not find a quadrilateral in the segmentation.")
            return

    # Order the corners consistently
    image_points = order_corners(corners)

    # Display the image with the detected corners
    for point in image_points:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)

    cv2.imshow('Detected Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        print("Error: Could not solve PnP problem.")
        return

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Output the 6D pose
    print("Rotation Matrix:")
    print(rotation_matrix)
    print("\nTranslation Vector:")
    print(translation_vector)

    # Pose visualization shifted to the center
    # Define the axes to be drawn from the center
    axis_length = tag_size * 0.5  # Adjust the axis length as needed
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
    image_pose = image.copy()
    corner = tuple(image_points_axes[0])  # Origin point at center of tag
    cv2.line(image_pose, corner, tuple(image_points_axes[1]), (0, 0, 255), 3)  # X-axis in red
    cv2.line(image_pose, corner, tuple(image_points_axes[2]), (0, 255, 0), 3)  # Y-axis in green
    cv2.line(image_pose, corner, tuple(image_points_axes[3]), (255, 0, 0), 3)  # Z-axis in blue

    # Show the image with pose visualization
    cv2.imshow('Pose Visualization Centered', image_pose)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
