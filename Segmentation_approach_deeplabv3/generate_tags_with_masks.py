import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from cv2 import aruco

# Configuration dictionary
config = {
    'num_images': 10000,
    'image_width': 640,
    'image_height': 480,
    'tag_family': 'tag36h11',
    'tag_size': 100,
    'output_dir': '../dataset_segmentation',
    'num_tags_per_image': (1, 3),  # Min and max number of tags per image
    'scale_factor_range': (0.5, 1.5)  # Scaling range for the tags
}

# Function to generate an AprilTag image
def generate_apriltag(tag_id, tag_family='tag36h11', tag_size=100):
    if tag_family == 'tag36h11':
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    else:
        print("Unknown tag family")
        return None
    marker_image = aruco.generateImageMarker(dictionary, tag_id, tag_size)
    return marker_image

# Function to create a random background
def create_random_background(width, height):
    color = np.random.randint(200, 256, size=(3,), dtype=np.uint8)
    background = np.full((height, width, 3), color, dtype=np.uint8)
    return background

# Function to rotate an image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    nW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
    nH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=255)
    return rotated

# Function to check for overlap between bounding boxes
def check_overlap(bbox, existing_bboxes):
    x1, y1, w1, h1 = bbox
    box1 = [x1, y1, x1 + w1, y1 + h1]
    for ex_bbox in existing_bboxes:
        x2, y2, w2, h2 = ex_bbox
        box2 = [x2, y2, x2 + w2, y2 + h2]
        if (box1[0] < box2[2] and box1[2] > box2[0] and
            box1[1] < box2[3] and box1[3] > box2[1]):
            return True
    return False

# Function to overlay the AprilTag onto the background and generate the mask
def overlay_apriltag(background_image, tag_image, existing_bboxes, scale_factor=1.0, max_attempts=50):
    h_bg, w_bg, _ = background_image.shape
    h_tag, w_tag = tag_image.shape
    new_w_tag, new_h_tag = int(w_tag * scale_factor), int(h_tag * scale_factor)
    tag_image_resized = cv2.resize(tag_image, (new_w_tag, new_h_tag))
    angle = random.uniform(-360, 360)
    tag_image_rotated = rotate_image(tag_image_resized, angle)
    mask = np.zeros((tag_image_rotated.shape[0], tag_image_rotated.shape[1]), dtype=np.uint8)
    mask[tag_image_rotated < 128] = 255

    attempt = 0
    while attempt < max_attempts:
        max_x, max_y = w_bg - tag_image_rotated.shape[1], h_bg - tag_image_rotated.shape[0]
        if max_x <= 0 or max_y <= 0:
            print("Tag image is larger than background image.")
            return None, None, None
        x, y = random.randint(0, max_x), random.randint(0, max_y)
        bbox = (x, y, tag_image_rotated.shape[1], tag_image_rotated.shape[0])

        if not check_overlap(bbox, existing_bboxes):
            roi = background_image[y:y+tag_image_rotated.shape[0], x:x+tag_image_rotated.shape[1]]
            tag_bgr = cv2.cvtColor(tag_image_rotated, cv2.COLOR_GRAY2BGR)
            _, tag_mask = cv2.threshold(tag_image_rotated, 254, 255, cv2.THRESH_BINARY_INV)
            tag_mask_inv = cv2.bitwise_not(tag_mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=tag_mask_inv)
            tag_fg = cv2.bitwise_and(tag_bgr, tag_bgr, mask=tag_mask)
            dst = cv2.add(img_bg, tag_fg)
            background_image[y:y+tag_image_rotated.shape[0], x:x+tag_image_rotated.shape[1]] = dst
            full_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
            full_mask[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
            return background_image, bbox, full_mask
        else:
            attempt += 1
    print("Failed to place tag without overlap after maximum attempts.")
    return None, None, None

# Function to create a single image in the dataset
def create_single_image(args):
    image_count, tag_ids, config = args
    bg_image = create_random_background(config['image_width'], config['image_height'])
    mask_image = np.zeros((config['image_height'], config['image_width']), dtype=np.uint8)
    num_tags_in_image = random.randint(config['num_tags_per_image'][0], config['num_tags_per_image'][1])
    existing_bboxes = []

    for _ in range(num_tags_in_image):
        tag_id = random.choice(tag_ids)
        scale_factor = random.uniform(config['scale_factor_range'][0], config['scale_factor_range'][1])
        tag_image = generate_apriltag(tag_id, config['tag_family'], config['tag_size'])
        if tag_image is None:
            continue
        result = overlay_apriltag(bg_image, tag_image, existing_bboxes, scale_factor=scale_factor)
        if result is None or result[0] is None:
            continue
        bg_image, bbox, tag_mask = result
        mask_image = cv2.bitwise_or(mask_image, tag_mask)
        existing_bboxes.append(bbox)

    images_dir = os.path.join(config['output_dir'], 'images')
    masks_dir = os.path.join(config['output_dir'], 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    image_filename = os.path.join(images_dir, f"image_{image_count}.jpg")
    mask_filename = os.path.join(masks_dir, f"mask_{image_count}.png")
    cv2.imwrite(image_filename, bg_image)
    cv2.imwrite(mask_filename, mask_image)

# Main function for multiprocessing
def main():
    tag_ids = list(range(0, 100))
    num_images = config['num_images']
    print("Generating images...")
    num_processes = min(cpu_count(), 8)
    pool = Pool(processes=num_processes)
    args = [(i, tag_ids, config) for i in range(num_images)]
    list(tqdm(pool.imap_unordered(create_single_image, args), total=num_images))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()