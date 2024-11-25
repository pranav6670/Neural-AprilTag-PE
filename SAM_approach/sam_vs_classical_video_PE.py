import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pyapriltags import Detector
import tqdm


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
    cv2.line(image, origin, tuple(image_points[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(image_points[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(image_points[3]), (255, 0, 0), 2)


# Function for video processing
def process_video(input_video_path, output_video_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h",
                  tag_size=0.1):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    detector = Detector(families='tag36h11')

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {input_video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    # Camera parameters
    fx_cam, fy_cam = 800, 800
    cx, cy = width / 2, height / 2
    camera_matrix = np.array([[fx_cam, 0, cx], [0, fy_cam, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    half_size = tag_size / 2.0
    object_points = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0]
    ], dtype=np.float32)

    with tqdm.tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx_cam, fy_cam, cx, cy),
                                      tag_size=tag_size)
            image_pose_apriltag = frame.copy()
            image_pose_sam = frame.copy()

            for result in results:
                tag_id = result.tag_id
                t_vector_apriltag = result.pose_t.flatten()
                r_matrix_apriltag = result.pose_R
                rotation_vector_apriltag, _ = cv2.Rodrigues(r_matrix_apriltag)

                draw_cube_on_tags(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix,
                                  dist_coeffs, tag_size)
                draw_pose_vectors(image_pose_apriltag, rotation_vector_apriltag, t_vector_apriltag, camera_matrix,
                                  dist_coeffs, tag_size)

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
                    draw_cube_on_tags(image_pose_sam, rotation_vector_sam, translation_vector_sam, camera_matrix,
                                      dist_coeffs, tag_size, color=(255, 0, 0))
                    draw_pose_vectors(image_pose_sam, rotation_vector_sam, translation_vector_sam, camera_matrix,
                                      dist_coeffs, tag_size)

            combined_frame = np.hstack((image_pose_sam, image_pose_apriltag))
            out.write(combined_frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Output video saved to: {output_video_path}")


# Run the video processing
input_video_path = "tags2.mp4"
output_video_path = "output_tags2_compare.mp4"
process_video(input_video_path, output_video_path)