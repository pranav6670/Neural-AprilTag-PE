# Filename: inference.py

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
import torch.nn as nn
import argparse

def load_model(model_path, device):
    """
    Loads the trained DeepLabv3+ model.
    Args:
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model onto.
    Returns:
        model (torch.nn.Module): The loaded model.
    """
    # Load the model architecture
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    model = models.deeplabv3_resnet101(weights=None)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image, device):
    """
    Preprocesses the input image for the model.
    Args:
        image (np.array): Input image.
        device (torch.device): Device to load the image onto.
    Returns:
        input_tensor (torch.Tensor): Preprocessed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    return input_tensor

def postprocess_output(output):
    """
    Postprocesses the model output to generate a binary mask.
    Args:
        output (torch.Tensor): Model output tensor.
    Returns:
        mask (np.array): Binary mask of detected AprilTags.
    """
    output_predictions = output.argmax(1)
    mask = output_predictions.byte().cpu().numpy()[0]
    return mask

def find_tag_corners(mask):
    """
    Finds the corners of detected tags in the mask.
    Args:
        mask (np.array): Binary mask of detected AprilTags.
    Returns:
        corners_list (list): List of corner points for each detected tag.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners_list = []
    for cnt in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            corners = np.squeeze(approx)
            corners_list.append(corners)
    return corners_list

def compute_pose(corners_list, camera_matrix, dist_coeffs, tag_size=0.162):
    """
    Computes the pose of the AprilTags using PnP.
    Args:
        corners_list (list): List of corner points for each detected tag.
        camera_matrix (np.array): Camera intrinsic matrix.
        dist_coeffs (np.array): Camera distortion coefficients.
        tag_size (float): Physical size of the tag (in meters).
    Returns:
        poses (list): List of rotation vectors and translation vectors.
    """
    # Define 3D points of the tag corners in the tag coordinate system
    half_size = tag_size / 2
    obj_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0],
    ])
    poses = []
    for corners in corners_list:
        # Ensure corners are in the correct order
        if corners.shape != (4, 2):
            continue
        img_points = np.array(corners, dtype=np.float32)
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if ret:
            poses.append((rvec, tvec))
    return poses

def draw_pose(image, corners_list, poses, camera_matrix, dist_coeffs):
    """
    Draws the detected tags and their poses on the image.
    Args:
        image (np.array): Original image.
        corners_list (list): List of corner points for each detected tag.
        poses (list): List of rotation vectors and translation vectors.
        camera_matrix (np.array): Camera intrinsic matrix.
        dist_coeffs (np.array): Camera distortion coefficients.
    Returns:
        image (np.array): Image with poses drawn.
    """
    axis = np.float32([
        [0, 0, 0],
        [0.05, 0, 0],
        [0, 0.05, 0],
        [0, 0, -0.05]
    ]).reshape(-1, 3)

    for corners, (rvec, tvec) in zip(corners_list, poses):
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw the tag border
        cv2.polylines(image, [np.int32(corners)], True, (0, 255, 0), 2)

        # Draw the coordinate axes
        corner = tuple(corners[0])
        image = cv2.line(image, corner, tuple(imgpts[1]), (0, 0, 255), 2)  # X axis in red
        image = cv2.line(image, corner, tuple(imgpts[2]), (0, 255, 0), 2)  # Y axis in green
        image = cv2.line(image, corner, tuple(imgpts[3]), (255, 0, 0), 2)  # Z axis in blue
    return image

def main():
    parser = argparse.ArgumentParser(description='AprilTag Segmentation and Pose Estimation')
    parser.add_argument('--model', type=str, default='best_deeplabv3_apriltag.pth', help='Path to the trained model')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--camera', action='store_true', help='Use live camera feed')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID')
    args = parser.parse_args()

    # Load camera intrinsic parameters (you need to calibrate your camera to get these)
    # Here we use dummy values; replace with your actual camera parameters
    fx = 600  # Focal length in pixels
    fy = 600
    cx = 320  # Principal point (image center)
    cy = 240
    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5)  # Assuming no lens distortion

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)

    if args.image:
        # Process a single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not read image {args.image}")
            return
        input_tensor = preprocess_image(image, device)
        with torch.no_grad():
            output = model(input_tensor)['out']
        mask = postprocess_output(output)

        # Find tag corners and compute pose
        corners_list = find_tag_corners(mask)
        poses = compute_pose(corners_list, camera_matrix, dist_coeffs)

        # Draw results
        result_image = draw_pose(image, corners_list, poses, camera_matrix, dist_coeffs)

        cv2.imshow('Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.camera:
        # Process live camera feed
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            input_tensor = preprocess_image(frame, device)
            with torch.no_grad():
                output = model(input_tensor)['out']
            mask = postprocess_output(output)

            # Find tag corners and compute pose
            corners_list = find_tag_corners(mask)
            poses = compute_pose(corners_list, camera_matrix, dist_coeffs)

            # Draw results
            result_frame = draw_pose(frame, corners_list, poses, camera_matrix, dist_coeffs)

            cv2.imshow('Live Pose Estimation', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Please provide an image path or use the --camera flag.")
        parser.print_help()

if __name__ == '__main__':
    main()
