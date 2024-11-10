import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import warnings
from pyapriltags import Detector
import time
import psutil
import tracemalloc
import threading
import pynvml

plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings("ignore")

# Function to order the corners consistently
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
mode = 'box'
points = []
user_interactions = 0
user_interaction_start = 0
user_interaction_end = 0

def select_prompt(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, mode, image_display, points, user_interactions, user_interaction_start, user_interaction_end
    if mode == 'box':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y
            user_interactions += 1
            if user_interactions == 1:
                user_interaction_start = time.time()
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
            if user_interactions > 0:
                user_interaction_end = time.time()
    elif mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select Prompt', image_display)
            points.append((x, y))
            user_interactions += 1
            if user_interactions == 1:
                user_interaction_start = time.time()
            user_interaction_end = time.time()

def monitor_resource_usage(interval, stop_event, resource_stats):
    process = psutil.Process()
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_mem_usage = mem_info.used / (1024 ** 2)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        energy_usage = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000  # in watts
        resource_stats['cpu_usage'].append(cpu_usage)
        resource_stats['gpu_mem_usage'].append(gpu_mem_usage)
        resource_stats['gpu_utilization'].append(gpu_util)
        resource_stats['energy_usage'].append(energy_usage)
        time.sleep(interval)

def main():
    global mode, image_display, points, fx, fy, drawing, user_interactions, user_interaction_start, user_interaction_end

    process = psutil.Process()
    tracemalloc.start()

    image_path = 'resized-images/IMG_8686.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return
    image_display = image.copy()

    print("Select prompt mode:\n1. Bounding Box\n2. Point(s)")
    mode_input = input("Enter 1 or 2: ")

    mode = 'box' if mode_input == '1' else 'point' if mode_input == '2' else None
    if mode is None:
        print("Invalid input.")
        return

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

    # Initialize data structures for resource tracking
    resource_stats_sam = {'cpu_usage': [], 'gpu_mem_usage': [], 'gpu_utilization': [], 'energy_usage': []}
    resource_stats_apriltag = {'cpu_usage': [], 'gpu_mem_usage': [], 'gpu_utilization': [], 'energy_usage': []}

    # ================= SAM Method =================
    start_time_sam = time.time()
    mem_before_sam = process.memory_info().rss / 1024 ** 2
    snapshot_before_sam = tracemalloc.take_snapshot()

    stop_event_sam = threading.Event()
    monitor_thread_sam = threading.Thread(
        target=monitor_resource_usage,
        args=(0.1, stop_event_sam, resource_stats_sam)
    )
    monitor_thread_sam.start()

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

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
        input_label = np.ones(len(points))
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

    mask = masks[0]
    stop_event_sam.set()
    monitor_thread_sam.join()

    end_time_sam = time.time()
    mem_after_sam = process.memory_info().rss / 1024 ** 2
    snapshot_after_sam = tracemalloc.take_snapshot()

    time_sam = end_time_sam - start_time_sam
    mem_usage_sam = mem_after_sam - mem_before_sam

    # Process segmentation to extract corners
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx_polygon) == 4:
        corners = approx_polygon.reshape(4, 2)
    else:
        print("Error: Could not find a quadrilateral in the segmentation.")
        return

    image_points_sam = order_corners(corners)

    # Define real-world coordinates for AprilTag corners
    tag_size = 0.1
    half_size = tag_size / 2.0
    object_points = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0]
    ], dtype=np.float32)

    # Camera parameters
    fx_cam, fy_cam = 800, 800
    cx, cy = image.shape[1] / 2, image.shape[0] / 2
    camera_matrix = np.array([
        [fx_cam, 0, cx],
        [0, fy_cam, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success_sam, rotation_vector_sam, translation_vector_sam = cv2.solvePnP(
        object_points,
        image_points_sam.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # ================= pyAprilTags Method =================
    start_time_apriltag = time.time()
    mem_before_apriltag = process.memory_info().rss / 1024 ** 2
    snapshot_before_apriltag = tracemalloc.take_snapshot()

    stop_event_apriltag = threading.Event()
    monitor_thread_apriltag = threading.Thread(
        target=monitor_resource_usage,
        args=(0.1, stop_event_apriltag, resource_stats_apriltag)
    )
    monitor_thread_apriltag.start()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = Detector(families='tag36h11')
    camera_params = (fx_cam, fy_cam, cx, cy)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    stop_event_apriltag.set()
    monitor_thread_apriltag.join()

    end_time_apriltag = time.time()
    mem_after_apriltag = process.memory_info().rss / 1024 ** 2
    snapshot_after_apriltag = tracemalloc.take_snapshot()

    time_apriltag = end_time_apriltag - start_time_apriltag
    mem_usage_apriltag = mem_after_apriltag - mem_before_apriltag

    if results:
        result = results[0]
        rotation_matrix_sam, _ = cv2.Rodrigues(rotation_vector_sam)
        rotation_matrix_apriltag = result.pose_R
        t_vector_apriltag = result.pose_t

        # Calculate differences between poses
        rotation_diff = rotation_matrix_sam - rotation_matrix_apriltag
        translation_diff = translation_vector_sam - t_vector_apriltag

        # Norm of differences
        rotation_diff_norm = np.linalg.norm(rotation_diff)
        translation_diff_norm = np.linalg.norm(translation_diff)

    # Calculate user interaction time
    user_interaction_time = user_interaction_end - user_interaction_start if user_interaction_start and user_interaction_end else 0

    # ================= Visualization Functions =================
    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha="center", va="bottom", fontsize=10,
                     color="black")

    def plot_execution_time(execution_times):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(execution_times.keys(), execution_times.values(), color=['#5DADE2', '#F1948A'], width=0.6)
        plt.title('Execution Time Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        add_labels(bars)
        plt.tight_layout()
        plt.savefig('execution_time_comparison.png', dpi=300)
        plt.show()

    def plot_memory_usage(memory_usage):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(memory_usage.keys(), memory_usage.values(), color=['#82E0AA', '#F7DC6F'], width=0.6)
        plt.title('Memory Usage Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Memory Usage (MB)', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        add_labels(bars)
        plt.tight_layout()
        plt.savefig('memory_usage_comparison.png', dpi=300)
        plt.show()

    def plot_cpu_gpu_usage(avg_cpu_usage, avg_gpu_usage, avg_gpu_mem_usage, avg_energy_usage):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(avg_cpu_usage.keys(), avg_cpu_usage.values(), color=['#AF7AC5', '#F0B27A'], width=0.6)
        plt.title('Average CPU Usage Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('CPU Usage (%)', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        add_labels(bars)
        plt.tight_layout()
        plt.savefig('cpu_usage_comparison.png', dpi=300)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        bars1 = ax1.bar(avg_gpu_mem_usage.keys(), avg_gpu_mem_usage.values(), color=['#76D7C4', '#F5B7B1'], width=0.6)
        ax1.set_title('Average GPU Memory Usage Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('GPU Memory Usage (MB)', fontsize=12)
        ax1.tick_params(axis='x', labelrotation=45)
        add_labels(bars1)

        bars2 = ax2.bar(avg_energy_usage.keys(), avg_energy_usage.values(), color=['#85C1E9', '#D7BDE2'], width=0.6)
        ax2.set_title('Average Energy Usage Comparison', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Energy Usage (Watts)', fontsize=12)
        ax2.tick_params(axis='x', labelrotation=45)
        add_labels(bars2)

        plt.tight_layout()
        plt.savefig('gpu_memory_energy_usage_comparison.png', dpi=300)
        plt.show()

    def plot_pose_diff_norms(pose_diff_norms):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(pose_diff_norms.keys(), pose_diff_norms.values(), color=['#58D68D', '#E59866'], width=0.6)
        plt.title('Pose Difference Norms', fontsize=16, fontweight='bold')
        plt.ylabel('Norm of Difference', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        add_labels(bars)
        plt.tight_layout()
        plt.savefig('pose_difference_norms.png', dpi=300)
        plt.show()

    def plot_reprojection_errors(errors):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(errors.keys(), errors.values(), color=['#5499C7', '#AF7AC5'], width=0.6)
        plt.title('Reprojection Error Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Reprojection Error', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        add_labels(bars)
        plt.tight_layout()
        plt.savefig('reprojection_error_comparison.png', dpi=300)
        plt.show()

    # ================= Call Visualization Functions =================
    avg_cpu_usage_sam = sum(resource_stats_sam['cpu_usage']) / len(resource_stats_sam['cpu_usage']) if resource_stats_sam['cpu_usage'] else 0
    avg_gpu_mem_usage_sam = sum(resource_stats_sam['gpu_mem_usage']) / len(resource_stats_sam['gpu_mem_usage']) if resource_stats_sam['gpu_mem_usage'] else 0
    avg_gpu_util_sam = sum(resource_stats_sam['gpu_utilization']) / len(resource_stats_sam['gpu_utilization']) if resource_stats_sam['gpu_utilization'] else 0
    avg_energy_usage_sam = sum(resource_stats_sam['energy_usage']) / len(resource_stats_sam['energy_usage']) if resource_stats_sam['energy_usage'] else 0

    avg_cpu_usage_apriltag = sum(resource_stats_apriltag['cpu_usage']) / len(resource_stats_apriltag['cpu_usage']) if resource_stats_apriltag['cpu_usage'] else 0
    avg_gpu_mem_usage_apriltag = sum(resource_stats_apriltag['gpu_mem_usage']) / len(resource_stats_apriltag['gpu_mem_usage']) if resource_stats_apriltag['gpu_mem_usage'] else 0
    avg_gpu_util_apriltag = sum(resource_stats_apriltag['gpu_utilization']) / len(resource_stats_apriltag['gpu_utilization']) if resource_stats_apriltag['gpu_utilization'] else 0
    avg_energy_usage_apriltag = sum(resource_stats_apriltag['energy_usage']) / len(resource_stats_apriltag['energy_usage']) if resource_stats_apriltag['energy_usage'] else 0

    plot_execution_time({'SAM': time_sam, 'pyAprilTags': time_apriltag})
    plot_memory_usage({'SAM': mem_usage_sam, 'pyAprilTags': mem_usage_apriltag})
    plot_cpu_gpu_usage({'SAM': avg_cpu_usage_sam, 'pyAprilTags': avg_cpu_usage_apriltag},
                       {'SAM': avg_gpu_util_sam, 'pyAprilTags': avg_gpu_util_apriltag},
                       {'SAM': avg_gpu_mem_usage_sam, 'pyAprilTags': avg_gpu_mem_usage_apriltag},
                       {'SAM': avg_energy_usage_sam, 'pyAprilTags': avg_energy_usage_apriltag})

    plot_pose_diff_norms({'Rotation': rotation_diff_norm, 'Translation': translation_diff_norm})
    # plot_reprojection_errors({'SAM': reprojection_error_sam, 'pyAprilTags': reprojection_error_apriltag})

    print(f"User Interaction Time: {user_interaction_time:.2f} seconds")

if __name__ == "__main__":
    main()
