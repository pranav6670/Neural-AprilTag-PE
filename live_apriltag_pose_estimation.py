import cv2
import numpy as np
from pyapriltags import Detector
import threading
import queue

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
    # Define cube points in 3D space
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
        cv2.line(frame, tuple(image_points[i]), tuple(image_points[j]), (0, 255, 255), 2)
        cv2.line(frame, tuple(image_points[i + 4]), tuple(image_points[j + 4]), (0, 255, 255), 2)
        cv2.line(frame, tuple(image_points[i]), tuple(image_points[i + 4]), (0, 255, 255), 2)

# Live pose visualization with multithreading and rotational overlay
def live_pose_visualization(camera_index=0, tag_size=0.1):
    # Camera parameters (adjust as per your camera setup)
    fx_cam, fy_cam = 800, 800  # Example focal lengths
    cx, cy = 640, 360  # Assuming a 1280x720 resolution
    camera_matrix = np.array([[fx_cam, 0, cx], [0, fy_cam, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Initialize AprilTags detector
    detector = Detector(families='tag36h11')

    # Queues for multithreading
    frame_queue = queue.Queue(maxsize=2)
    output_queue = queue.Queue(maxsize=2)

    # Start the processing thread
    processing_thread = threading.Thread(target=process_frames, args=(frame_queue, output_queue, detector, camera_matrix, dist_coeffs, tag_size))
    processing_thread.daemon = True
    processing_thread.start()

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (1280, 720))

        # Add frame to the queue
        if not frame_queue.full():
            frame_queue.put(frame)

        # Get processed results
        if not output_queue.empty():
            frame, results = output_queue.get()

            for result in results:
                tag_corners = result.corners
                ordered_corners = order_corners(tag_corners)
                tag_id = result.tag_id
                t_vector = result.pose_t.flatten()
                r_matrix = result.pose_R
                rotation_vector, _ = cv2.Rodrigues(r_matrix)

                # Draw detected tag ID
                top_right = tuple(map(int, ordered_corners[1]))
                cv2.putText(
                    frame,
                    f"ID: {tag_id}",
                    (top_right[0] - 30, top_right[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

                # Draw pose axes for visualization
                axis_3D_points = np.float32([
                    [0, 0, 0], [tag_size * 0.5, 0, 0],  # X-axis (red)
                    [0, tag_size * 0.5, 0],             # Y-axis (green)
                    [0, 0, -tag_size * 0.5]             # Z-axis (blue)
                ])
                image_points_axes, _ = cv2.projectPoints(
                    axis_3D_points, rotation_vector, t_vector,
                    camera_matrix, dist_coeffs
                )
                image_points_axes = image_points_axes.reshape(-1, 2).astype(int)

                # Draw axes
                corner = tuple(image_points_axes[0])
                cv2.line(frame, corner, tuple(image_points_axes[1]), (0, 0, 255), 3)  # X-axis in red
                cv2.line(frame, corner, tuple(image_points_axes[2]), (0, 255, 0), 3)  # Y-axis in green
                cv2.line(frame, corner, tuple(image_points_axes[3]), (255, 0, 0), 3)  # Z-axis in blue

                # Display pose vector
                t_str = f"T: {t_vector[0]:.2f}, {t_vector[1]:.2f}, {t_vector[2]:.2f}"
                cv2.putText(frame, t_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw 3D cube for rotation visualization
                draw_cube(frame, camera_matrix, dist_coeffs, rotation_vector, t_vector, tag_size)

            # Show the output frame
            cv2.imshow("Live Pose Visualization with Rotation", frame)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    frame_queue.put(None)  # Signal the processing thread to exit
    processing_thread.join()

# Run live pose visualization
live_pose_visualization()
