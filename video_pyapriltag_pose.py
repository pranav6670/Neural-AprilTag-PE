import cv2
import numpy as np
from pyapriltags import Detector
import threading
import queue
from tqdm import tqdm  # Import tqdm for the progress bar

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

# Worker thread for detecting AprilTags
def process_frames(frame_queue, output_queue, detector, camera_matrix, dist_coeffs, tag_size):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        results = detector.detect(gray, estimate_tag_pose=True, camera_params=(camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]), tag_size=tag_size)

        # Store results in the output queue
        output_queue.put((frame, results))

# Draw a 3D cube on the tag for rotation visualization
def draw_cube(frame, camera_matrix, dist_coeffs, rotation_vector, t_vector, tag_size):
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
        cv2.line(frame, tuple(image_points[i]), tuple(image_points[j]), (0, 255, 255), 2)
        cv2.line(frame, tuple(image_points[i + 4]), tuple(image_points[j + 4]), (0, 255, 255), 2)
        cv2.line(frame, tuple(image_points[i]), tuple(image_points[i + 4]), (0, 255, 255), 2)

# Video processing with tqdm for progress tracking
def process_video(input_video_path, output_video_path, tag_size=0.1):
    fx_cam, fy_cam = 800, 800
    cx, cy = 640, 360
    camera_matrix = np.array([[fx_cam, 0, cx], [0, fy_cam, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    detector = Detector(families='tag36h11')

    frame_queue = queue.Queue(maxsize=2)
    output_queue = queue.Queue(maxsize=2)

    processing_thread = threading.Thread(target=process_frames, args=(frame_queue, output_queue, detector, camera_matrix, dist_coeffs, tag_size))
    processing_thread.daemon = True
    processing_thread.start()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open the input video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {input_video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Processing complete.")
                break

            frame_count += 1
            pbar.update(1)  # Update tqdm progress bar

            if not frame_queue.full():
                frame_queue.put(frame)

            if not output_queue.empty():
                frame, results = output_queue.get()

                for result in results:
                    tag_corners = result.corners
                    ordered_corners = order_corners(tag_corners)
                    tag_id = result.tag_id
                    t_vector = result.pose_t.flatten()
                    r_matrix = result.pose_R
                    rotation_vector, _ = cv2.Rodrigues(r_matrix)

                    top_right = tuple(map(int, ordered_corners[1]))
                    cv2.putText(frame, f"ID: {tag_id}", (top_right[0] - 30, top_right[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    axis_3D_points = np.float32([
                        [0, 0, 0], [tag_size * 0.5, 0, 0], [0, tag_size * 0.5, 0],
                        [0, 0, -tag_size * 0.5]
                    ])
                    image_points_axes, _ = cv2.projectPoints(axis_3D_points, rotation_vector, t_vector, camera_matrix, dist_coeffs)
                    image_points_axes = image_points_axes.reshape(-1, 2).astype(int)
                    corner = tuple(image_points_axes[0])
                    cv2.line(frame, corner, tuple(image_points_axes[1]), (0, 0, 255), 3)
                    cv2.line(frame, corner, tuple(image_points_axes[2]), (0, 255, 0), 3)
                    cv2.line(frame, corner, tuple(image_points_axes[3]), (255, 0, 0), 3)

                    draw_cube(frame, camera_matrix, dist_coeffs, rotation_vector, t_vector, tag_size)

                out.write(frame)

    cap.release()
    out.release()
    frame_queue.put(None)
    processing_thread.join()

    print(f"Output saved to: {output_video_path}")

# Run video processing with tqdm
input_video_path = 'SAM_approach/tags2.mp4'
output_video_path = 'tags2_out.mp4'
process_video(input_video_path, output_video_path)
