import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from pyapriltags import Detector
import time
import psutil
import tracemalloc
import threading
import pynvml
import warnings
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings("ignore")

# Set seaborn theme for better aesthetics
sns.set_theme(style="whitegrid")

# Increase default font sizes for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14
})


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


def detect_tags_with_comparison(image_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Initialize data structures for resource tracking
    resource_stats_auto_sam = {'cpu_usage': [], 'gpu_mem_usage': [], 'gpu_utilization': [], 'energy_usage': []}
    resource_stats_apriltag = {'cpu_usage': [], 'gpu_mem_usage': [], 'gpu_utilization': [], 'energy_usage': []}

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Prepare common variables
    process = psutil.Process()
    tracemalloc.start()
    tag_size = 0.1  # Example: 0.1 meters
    half_size = tag_size / 2.0
    object_points = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0]
    ], dtype=np.float32)

    fx_cam, fy_cam = 800, 800  # Focal lengths
    cx, cy = image.shape[1] / 2, image.shape[0] / 2  # Principal point
    camera_matrix = np.array([
        [fx_cam, 0, cx],
        [0, fy_cam, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # ==============================================================
    # ================= SAM Automatic Segmentation Method ==========
    # ==============================================================
    start_time_sam_auto_total = time.time()
    mem_before_sam_auto = process.memory_info().rss / 1024 ** 2
    snapshot_before_sam_auto = tracemalloc.take_snapshot()

    stop_event_sam_auto = threading.Event()
    monitor_thread_sam_auto = threading.Thread(
        target=monitor_resource_usage,
        args=(0.1, stop_event_sam_auto, resource_stats_auto_sam)
    )
    monitor_thread_sam_auto.start()

    # Initialize SAM
    sam_auto = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam_auto)

    # Generate masks automatically
    start_time_sam_auto_segmentation = time.time()
    masks = mask_generator.generate(image)
    end_time_sam_auto_segmentation = time.time()
    segmentation_time_sam_auto = end_time_sam_auto_segmentation - start_time_sam_auto_segmentation

    stop_event_sam_auto.set()
    monitor_thread_sam_auto.join()

    end_time_sam_auto_total = time.time()
    mem_after_sam_auto = process.memory_info().rss / 1024 ** 2
    snapshot_after_sam_auto = tracemalloc.take_snapshot()

    total_time_sam_auto = end_time_sam_auto_total - start_time_sam_auto_total
    mem_usage_sam_auto = mem_after_sam_auto - mem_before_sam_auto

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
        border = gray[y:y + h, x:x + w]

        # Check if the border is predominantly black
        border_pixels = border.flatten()
        black_pixel_ratio = np.sum(border_pixels < 50) / len(border_pixels)  # Adjust threshold as needed

        if black_pixel_ratio > 0.6:  # Adjust threshold based on your dataset
            selected_mask = mask
            break

    if selected_mask is None:
        print("No suitable tag with a black border was found.")
        return

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
    image_points_sam_auto = order_corners(corners)

    # Pose Estimation using SAM Auto
    start_time_sam_auto_pose = time.time()
    success_sam_auto, rotation_vector_sam_auto, translation_vector_sam_auto = cv2.solvePnP(
        object_points,
        image_points_sam_auto.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    end_time_sam_auto_pose = time.time()
    pose_estimation_time_sam_auto = end_time_sam_auto_pose - start_time_sam_auto_pose

    if not success_sam_auto:
        print("Error: Could not solve PnP problem with SAM automatic method.")
        return

    # Convert rotation vector to rotation matrix
    rotation_matrix_sam_auto, _ = cv2.Rodrigues(rotation_vector_sam_auto)

    # ==============================================================
    # ================= pyAprilTags Method =========================
    # ==============================================================
    start_time_apriltag_total = time.time()
    mem_before_apriltag = process.memory_info().rss / 1024 ** 2
    snapshot_before_apriltag = tracemalloc.take_snapshot()

    stop_event_apriltag = threading.Event()
    monitor_thread_apriltag = threading.Thread(
        target=monitor_resource_usage,
        args=(0.1, stop_event_apriltag, resource_stats_apriltag)
    )
    monitor_thread_apriltag.start()

    # Convert image to grayscale for pyAprilTags
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the AprilTag detector
    detector = Detector(families='tag36h11')

    # Camera parameters for pyAprilTags
    camera_params = (fx_cam, fy_cam, cx, cy)

    # Detect tags in the image
    start_time_apriltag_detection = time.time()
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    end_time_apriltag_detection = time.time()
    detection_time_apriltag = end_time_apriltag_detection - start_time_apriltag_detection

    stop_event_apriltag.set()
    monitor_thread_apriltag.join()

    end_time_apriltag_total = time.time()
    mem_after_apriltag = process.memory_info().rss / 1024 ** 2
    snapshot_after_apriltag = tracemalloc.take_snapshot()

    total_time_apriltag = end_time_apriltag_total - start_time_apriltag_total
    mem_usage_apriltag = mem_after_apriltag - mem_before_apriltag

    if not results:
        print("No tags detected by pyAprilTags.")
        return

    # Assume the first detected tag
    result = results[0]

    # Get pose estimation from pyAprilTags
    t_vector_apriltag = result.pose_t  # Translation vector
    rotation_vector_apriltag, _ = cv2.Rodrigues(result.pose_R)

    # ==============================================================
    # ==================== Pose Comparison =========================
    # ==============================================================
    # Calculate differences between poses
    rotation_matrix_sam_auto, _ = cv2.Rodrigues(rotation_vector_sam_auto)
    rotation_matrix_apriltag, _ = cv2.Rodrigues(rotation_vector_apriltag)

    rotation_diff_sam_auto = rotation_matrix_sam_auto - rotation_matrix_apriltag
    translation_diff_sam_auto = translation_vector_sam_auto - t_vector_apriltag

    # Norm of differences
    rotation_diff_norm_sam_auto = np.linalg.norm(rotation_diff_sam_auto)
    translation_diff_norm_sam_auto = np.linalg.norm(translation_diff_sam_auto)

    # ==============================================================
    # ==================== Reprojection Error ======================
    # ==============================================================
    def calculate_reprojection_error(image_points, rotation_vector, translation_vector):
        projected_points, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        error = cv2.norm(image_points, projected_points, cv2.NORM_L2) / len(projected_points)
        return error

    reprojection_error_sam_auto = calculate_reprojection_error(image_points_sam_auto, rotation_vector_sam_auto,
                                                               translation_vector_sam_auto)
    reprojection_error_apriltag = result.hamming  # Hamming distance as an error metric (lower is better)

    # ==============================================================
    # ==================== Visualization Functions =================
    # ==============================================================
    def add_labels(ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # Adjusted for better visibility
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    def plot_execution_time(execution_times, segmentation_times, pose_times):
        methods = list(execution_times.keys())
        total_times = list(execution_times.values())
        segmentation_times = [segmentation_times[method] for method in methods]
        pose_times = [pose_times[method] for method in methods]
        other_times = [total - seg - pose for total, seg, pose in zip(total_times, segmentation_times, pose_times)]

        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.25

        indices = np.arange(len(methods))
        p1 = ax.bar(indices, segmentation_times, bar_width, label='Segmentation Time')
        p2 = ax.bar(indices, pose_times, bar_width, bottom=segmentation_times, label='Pose Estimation Time')
        p3 = ax.bar(indices, other_times, bar_width, bottom=[i + j for i, j in zip(segmentation_times, pose_times)],
                    label='Other Time')

        ax.set_title('Execution Time Comparison', fontsize=22, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=18)
        ax.set_xticks(indices)
        ax.set_xticklabels(methods, rotation=45, fontsize=16)
        ax.legend(fontsize=16)

        # Add labels
        for rect in p1 + p2 + p3:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),
                            xytext=(0, 0),  # No offset
                            textcoords="offset points",
                            ha='center', va='center', fontsize=14, color='white')

        plt.tight_layout()
        plt.savefig('execution_time_comparison.png', dpi=300)
        plt.show()

    def plot_memory_usage(memory_usage):
        methods = list(memory_usage.keys())
        mem_usages = list(memory_usage.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.25

        rects = ax.bar(methods, mem_usages, bar_width, color=sns.color_palette("Set2"))
        ax.set_title('Memory Usage Comparison', fontsize=22, fontweight='bold')
        ax.set_ylabel('Memory Usage (MB)', fontsize=18)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, fontsize=16)

        add_labels(ax, rects)

        plt.tight_layout()
        plt.savefig('memory_usage_comparison.png', dpi=300)
        plt.show()

    def plot_cpu_gpu_usage(cpu_usages, gpu_mem_usages, gpu_utils, energy_usages):
        methods = list(cpu_usages.keys())

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        bar_width = 0.25

        # CPU Usage
        rects1 = axs[0, 0].bar(methods, [cpu_usages[method] for method in methods], bar_width,
                               color=sns.color_palette("Set1"))
        axs[0, 0].set_title('Average CPU Usage (%)', fontsize=20)
        axs[0, 0].set_xticks(range(len(methods)))
        axs[0, 0].set_xticklabels(methods, rotation=45, fontsize=16)
        axs[0, 0].set_ylabel('CPU Usage (%)', fontsize=18)
        add_labels(axs[0, 0], rects1)

        # GPU Memory Usage
        rects2 = axs[0, 1].bar(methods, [gpu_mem_usages[method] for method in methods], bar_width,
                               color=sns.color_palette("Set2"))
        axs[0, 1].set_title('Average GPU Memory Usage (MB)', fontsize=20)
        axs[0, 1].set_xticks(range(len(methods)))
        axs[0, 1].set_xticklabels(methods, rotation=45, fontsize=16)
        axs[0, 1].set_ylabel('GPU Memory Usage (MB)', fontsize=18)
        add_labels(axs[0, 1], rects2)

        # GPU Utilization
        rects3 = axs[1, 0].bar(methods, [gpu_utils[method] for method in methods], bar_width,
                               color=sns.color_palette("Set3"))
        axs[1, 0].set_title('Average GPU Utilization (%)', fontsize=20)
        axs[1, 0].set_xticks(range(len(methods)))
        axs[1, 0].set_xticklabels(methods, rotation=45, fontsize=16)
        axs[1, 0].set_ylabel('GPU Utilization (%)', fontsize=18)
        add_labels(axs[1, 0], rects3)

        # Energy Usage
        rects4 = axs[1, 1].bar(methods, [energy_usages[method] for method in methods], bar_width,
                               color=sns.color_palette("Set1"))
        axs[1, 1].set_title('Average Energy Usage (Watts)', fontsize=20)
        axs[1, 1].set_xticks(range(len(methods)))
        axs[1, 1].set_xticklabels(methods, rotation=45, fontsize=16)
        axs[1, 1].set_ylabel('Energy Usage (Watts)', fontsize=18)
        add_labels(axs[1, 1], rects4)

        plt.tight_layout()
        plt.savefig('resource_usage_comparison.png', dpi=300)
        plt.show()

    def plot_pose_diff_norms(rotation_diffs, translation_diffs):
        methods = list(rotation_diffs.keys())

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        bar_width = 0.25

        # Rotation Difference Norms
        rects1 = axs[0].bar(methods, [rotation_diffs[method] for method in methods], bar_width,
                            color=sns.color_palette("Set2"))
        axs[0].set_title('Rotation Difference Norms', fontsize=20)
        axs[0].set_ylabel('Norm of Rotation Difference', fontsize=18)
        axs[0].set_xticks(range(len(methods)))
        axs[0].set_xticklabels(methods, rotation=45, fontsize=16)
        add_labels(axs[0], rects1)

        # Translation Difference Norms
        rects2 = axs[1].bar(methods, [translation_diffs[method] for method in methods], bar_width,
                            color=sns.color_palette("Set3"))
        axs[1].set_title('Translation Difference Norms', fontsize=20)
        axs[1].set_ylabel('Norm of Translation Difference', fontsize=18)
        axs[1].set_xticks(range(len(methods)))
        axs[1].set_xticklabels(methods, rotation=45, fontsize=16)
        add_labels(axs[1], rects2)

        plt.tight_layout()
        plt.savefig('pose_difference_norms.png', dpi=300)
        plt.show()

    def plot_reprojection_errors(errors):
        methods = list(errors.keys())
        reproj_errors = list(errors.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.25

        rects = ax.bar(methods, reproj_errors, bar_width, color=sns.color_palette("Set1"))
        ax.set_title('Reprojection Error Comparison', fontsize=22, fontweight='bold')
        ax.set_ylabel('Reprojection Error', fontsize=18)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, fontsize=16)

        add_labels(ax, rects)

        plt.tight_layout()
        plt.savefig('reprojection_error_comparison.png', dpi=300)
        plt.show()

    # ==============================================================
    # ==================== Call Visualization Functions ============
    # ==============================================================
    # Calculate average resource usages
    avg_cpu_usage_sam_auto = np.mean(resource_stats_auto_sam['cpu_usage']) if resource_stats_auto_sam[
        'cpu_usage'] else 0
    avg_gpu_mem_usage_sam_auto = np.mean(resource_stats_auto_sam['gpu_mem_usage']) if resource_stats_auto_sam[
        'gpu_mem_usage'] else 0
    avg_gpu_util_sam_auto = np.mean(resource_stats_auto_sam['gpu_utilization']) if resource_stats_auto_sam[
        'gpu_utilization'] else 0
    avg_energy_usage_sam_auto = np.mean(resource_stats_auto_sam['energy_usage']) if resource_stats_auto_sam[
        'energy_usage'] else 0

    avg_cpu_usage_apriltag = np.mean(resource_stats_apriltag['cpu_usage']) if resource_stats_apriltag[
        'cpu_usage'] else 0
    avg_gpu_mem_usage_apriltag = np.mean(resource_stats_apriltag['gpu_mem_usage']) if resource_stats_apriltag[
        'gpu_mem_usage'] else 0
    avg_gpu_util_apriltag = np.mean(resource_stats_apriltag['gpu_utilization']) if resource_stats_apriltag[
        'gpu_utilization'] else 0
    avg_energy_usage_apriltag = np.mean(resource_stats_apriltag['energy_usage']) if resource_stats_apriltag[
        'energy_usage'] else 0

    # Execution times
    execution_times = {
        'SAM Auto': total_time_sam_auto,
        'pyAprilTags': total_time_apriltag
    }
    segmentation_times = {
        'SAM Auto': segmentation_time_sam_auto,
        'pyAprilTags': detection_time_apriltag
    }
    pose_times = {
        'SAM Auto': pose_estimation_time_sam_auto,
        'pyAprilTags': 0  # Included in detection time
    }

    plot_execution_time(execution_times, segmentation_times, pose_times)

    # Memory usage
    memory_usage = {
        'SAM Auto': mem_usage_sam_auto,
        'pyAprilTags': mem_usage_apriltag
    }
    plot_memory_usage(memory_usage)

    # CPU and GPU usage
    cpu_usages = {
        'SAM Auto': avg_cpu_usage_sam_auto,
        'pyAprilTags': avg_cpu_usage_apriltag
    }
    gpu_mem_usages = {
        'SAM Auto': avg_gpu_mem_usage_sam_auto,
        'pyAprilTags': avg_gpu_mem_usage_apriltag
    }
    gpu_utils = {
        'SAM Auto': avg_gpu_util_sam_auto,
        'pyAprilTags': avg_gpu_util_apriltag
    }
    energy_usages = {
        'SAM Auto': avg_energy_usage_sam_auto,
        'pyAprilTags': avg_energy_usage_apriltag
    }
    plot_cpu_gpu_usage(cpu_usages, gpu_mem_usages, gpu_utils, energy_usages)

    # Pose difference norms
    rotation_diffs = {
        'SAM Auto': rotation_diff_norm_sam_auto
    }
    translation_diffs = {
        'SAM Auto': translation_diff_norm_sam_auto
    }
    plot_pose_diff_norms(rotation_diffs, translation_diffs)

    # Reprojection errors
    reprojection_errors = {
        'SAM Auto': reprojection_error_sam_auto,
        'pyAprilTags': reprojection_error_apriltag
    }
    plot_reprojection_errors(reprojection_errors)

    # Output the pose differences
    print(f"\nNorm of Rotation Matrix Difference (SAM Auto): {rotation_diff_norm_sam_auto}")
    print(f"Norm of Translation Vector Difference (SAM Auto): {translation_diff_norm_sam_auto}")


# Call the function
image_path = 'tags4.jpg'  # Replace with your image path
detect_tags_with_comparison(image_path)
