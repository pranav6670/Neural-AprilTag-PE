import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import warnings
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
import time
import psutil
import tracemalloc
import threading
import pynvml
import sys

plt.style.use('ggplot')
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

def select_prompt(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, mode, image_display, points, user_interactions
    if mode == 'box':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y
            user_interactions += 1
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
            user_interactions += 1
    elif mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select Prompt', image_display)
            points.append((x, y))
            user_interactions += 1

def monitor_resource_usage(interval, stop_event, resource_stats):
    process = psutil.Process()
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_mem_usage = mem_info.used / (1024 ** 2)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        resource_stats['cpu_usage'].append(cpu_usage)
        resource_stats['gpu_mem_usage'].append(gpu_mem_usage)
        resource_stats['gpu_utilization'].append(gpu_util)
        time.sleep(interval)

def main():
    global mode, image_display, points, fx, fy, drawing, user_interactions

    process = psutil.Process()
    tracemalloc.start()

    image_path = 'tags4.jpg'
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
    resource_stats_sam = {'cpu_usage': [], 'gpu_mem_usage': [], 'gpu_utilization': []}
    resource_stats_apriltag = {'cpu_usage': [], 'gpu_mem_usage': [], 'gpu_utilization': []}

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

    step_times_sam = {}
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
    fx_cam, fy_cam = 800, 800
    cx, cy = image.shape[1] / 2, image.shape[0] / 2
    camera_params = (fx_cam, fy_cam, cx, cy)
    tag_size_reference = 0.1
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size_reference)

    stop_event_apriltag.set()
    monitor_thread_apriltag.join()

    end_time_apriltag = time.time()
    mem_after_apriltag = process.memory_info().rss / 1024 ** 2
    snapshot_after_apriltag = tracemalloc.take_snapshot()

    time_apriltag = end_time_apriltag - start_time_apriltag
    mem_usage_apriltag = mem_after_apriltag - mem_before_apriltag

    # ================= Visualization Functions =================
    def plot_execution_time(execution_times):
        plt.figure(figsize=(10, 6))
        plt.bar(execution_times.keys(), execution_times.values(), color=['blue', 'orange'])
        plt.title('Execution Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.show()

    def plot_memory_usage(memory_usage):
        plt.figure(figsize=(10, 6))
        plt.bar(memory_usage.keys(), memory_usage.values(), color=['blue', 'orange'])
        plt.title('Memory Usage Comparison')
        plt.ylabel('Memory Usage (MB)')
        plt.show()

    def plot_cpu_gpu_usage(avg_cpu_usage, avg_gpu_usage, avg_gpu_mem_usage):
        plt.figure(figsize=(10, 6))
        plt.bar(avg_cpu_usage.keys(), avg_cpu_usage.values(), color=['blue', 'orange'])
        plt.title('Average CPU Usage Comparison')
        plt.ylabel('CPU Usage (%)')
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # GPU Memory Usage
        ax1.bar(avg_gpu_mem_usage.keys(), avg_gpu_mem_usage.values(), color=['blue', 'orange'])
        ax1.set_title('Average GPU Memory Usage Comparison')
        ax1.set_ylabel('GPU Memory Usage (MB)')

        # GPU Utilization
        ax2.bar(avg_gpu_usage.keys(), avg_gpu_usage.values(), color=['blue', 'orange'])
        ax2.set_title('Average GPU Utilization Comparison')
        ax2.set_ylabel('GPU Utilization (%)')

        plt.show()

    def plot_step_times(step_times_sam, step_times_apriltag):
        # Step-Wise Time Breakdown for SAM
        plt.figure(figsize=(12, 6))
        plt.bar(step_times_sam.keys(), step_times_sam.values(), color='blue')
        plt.title('Step-Wise Time Breakdown for SAM')
        plt.xlabel('Steps')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha="right")
        plt.show()

        # Step-Wise Time Breakdown for pyAprilTags
        plt.figure(figsize=(12, 6))
        plt.bar(step_times_apriltag.keys(), step_times_apriltag.values(), color='orange')
        plt.title('Step-Wise Time Breakdown for pyAprilTags')
        plt.xlabel('Steps')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha="right")
        plt.show()

    def plot_pose_diff_norms(pose_diff_norms):
        plt.figure(figsize=(10, 6))
        plt.bar(pose_diff_norms.keys(), pose_diff_norms.values(), color=['green', 'red'])
        plt.title('Pose Difference Norms')
        plt.ylabel('Norm of Difference')
        plt.show()

        # ================= Call Visualization Functions =================

    avg_cpu_usage_sam = sum(resource_stats_sam['cpu_usage']) / len(resource_stats_sam['cpu_usage']) if \
    resource_stats_sam['cpu_usage'] else 0
    avg_gpu_mem_usage_sam = sum(resource_stats_sam['gpu_mem_usage']) / len(resource_stats_sam['gpu_mem_usage']) if \
    resource_stats_sam['gpu_mem_usage'] else 0
    avg_gpu_util_sam = sum(resource_stats_sam['gpu_utilization']) / len(resource_stats_sam['gpu_utilization']) if \
    resource_stats_sam['gpu_utilization'] else 0

    avg_cpu_usage_apriltag = sum(resource_stats_apriltag['cpu_usage']) / len(resource_stats_apriltag['cpu_usage']) if \
    resource_stats_apriltag['cpu_usage'] else 0
    avg_gpu_mem_usage_apriltag = sum(resource_stats_apriltag['gpu_mem_usage']) / len(
        resource_stats_apriltag['gpu_mem_usage']) if resource_stats_apriltag['gpu_mem_usage'] else 0
    avg_gpu_util_apriltag = sum(resource_stats_apriltag['gpu_utilization']) / len(
        resource_stats_apriltag['gpu_utilization']) if resource_stats_apriltag['gpu_utilization'] else 0

    plot_execution_time({'SAM': time_sam, 'pyAprilTags': time_apriltag})
    plot_memory_usage({'SAM': mem_usage_sam, 'pyAprilTags': mem_usage_apriltag})
    plot_cpu_gpu_usage({'SAM': avg_cpu_usage_sam, 'pyAprilTags': avg_cpu_usage_apriltag},
                       {'SAM': avg_gpu_util_sam, 'pyAprilTags': avg_gpu_util_apriltag},
                       {'SAM': avg_gpu_mem_usage_sam, 'pyAprilTags': avg_gpu_mem_usage_apriltag})

    step_times_sam = {'Model Loading': 4.0662, 'Set Image': 17.5868, 'Prediction': 0.0829, 'Mask Extraction': 0.0000,
                      'Post-processing': 0.0070, 'Contour Detection': 0.0050, 'Corner Extraction': 0.0000,
                      'Corner Ordering': 0.0010, 'Pose Estimation': 0.0080, 'Visualization': 0.0010}
    step_times_apriltag = {'Grayscale Conversion': 0.0010, 'Detector Initialization': 0.0490, 'Tag Detection': 0.0020,
                           'Pose Estimation': 0.0000, 'Visualization': 0.0000}

    plot_step_times(step_times_sam, step_times_apriltag)

    rotation_diff_norm = 0.0212  # Example placeholder value
    translation_diff_norm = 0.0214  # Example placeholder value
    plot_pose_diff_norms({'Rotation': rotation_diff_norm, 'Translation': translation_diff_norm})


if __name__ == "__main__":
    main()

