import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from cv2 import aruco

# Folder path for the background images
background_folder = '../backgrounds/'

# Configuration dictionary
config = {
    'num_images': 100,
    'image_width': 640,
    'image_height': 480,
    'tag_family': 'tag36h11',
    'tag_size': 100,
    'output_dir': '../dataset_segmentation_warp1',
    'num_tags_per_image': (1, 1),  # Min and max number of tags per image
    'scale_factor_range': (0.5, 1.5)  # Scaling range for the tags
}


# Function to warp the tag image with four corners intact
def warp_tag_image(tag_image, intensity=0.2):
    if tag_image is None or tag_image.size == 0:
        return None
    h, w = tag_image.shape[:2]

    src_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    delta_x = intensity * w
    delta_y = intensity * h
    dst_points = np.float32([
        [random.uniform(0, delta_x), random.uniform(0, delta_y)],
        [w - 1 - random.uniform(0, delta_x), random.uniform(0, delta_y)],
        [w - 1 - random.uniform(0, delta_x), h - 1 - random.uniform(0, delta_y)],
        [random.uniform(0, delta_x), h - 1 - random.uniform(0, delta_y)]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_tag = cv2.warpPerspective(tag_image, matrix, (w, h), borderValue=255)
    return warped_tag


def generate_apriltag(tag_id, tag_family='tag36h11', tag_size=100, border_size_ratio=0.25):
    """
    Generate an AprilTag image with a proper quiet zone (white border) and solid black-and-white values.
    """
    # Validate tag family
    if tag_family == 'tag36h11':
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    else:
        raise ValueError("Unknown tag family: Supported families include 'tag36h11'.")

    # Generate the inner tag grid
    inner_tag = aruco.generateImageMarker(dictionary, tag_id, tag_size)

    # Force binary values (black = 0, white = 255)
    inner_tag_binary = (inner_tag > 127).astype(np.uint8) * 255

    # Calculate border size
    border_size = int(tag_size * border_size_ratio)

    # Add a solid white border (quiet zone) around the tag
    tag_with_border = cv2.copyMakeBorder(
        inner_tag_binary,
        border_size, border_size, border_size, border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=255  # White for the border
    )

    return tag_with_border




# Function to load a random background image from your folder
def load_random_background():
    background_files = os.listdir(background_folder)
    random_background_path = os.path.join(background_folder, random.choice(background_files))
    background_image = cv2.imread(random_background_path)
    if background_image is None:
        print(f"Could not load background image: {random_background_path}")
        return None
    return background_image


# Function to rotate an image
def rotate_image(image, angle):
    if image is None or image.size == 0:
        return None
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    nW = int((h * abs(M[0, 1])) + (w * abs(M[0, 0])))
    nH = int((h * abs(M[0, 0])) + (w * abs(M[0, 1])))
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


def overlay_apriltag(background_image, tag_image, existing_bboxes, scale_factor=1.0, max_attempts=50):
    """
    Overlay the AprilTag onto the background, ensuring solid black and white colors with no transparency.
    """
    if background_image is None or tag_image is None:
        return None, None, None

    h_bg, w_bg, _ = background_image.shape
    h_tag, w_tag = tag_image.shape

    # Resize tag image
    new_w_tag, new_h_tag = int(w_tag * scale_factor), int(h_tag * scale_factor)
    if new_w_tag <= 0 or new_h_tag <= 0:
        return None, None, None
    tag_image_resized = cv2.resize(tag_image, (new_w_tag, new_h_tag), interpolation=cv2.INTER_AREA)

    # Warp tag image
    tag_image_warped = warp_tag_image(tag_image_resized, intensity=0.2)

    # Rotate tag image
    angle = random.uniform(-45, 45)
    tag_image_rotated = rotate_image(tag_image_warped, angle)
    if tag_image_rotated is None or tag_image_rotated.size == 0:
        return None, None, None

    # Explicitly force the tag image to binary (solid black and white)
    _, tag_image_binary = cv2.threshold(tag_image_rotated, 127, 255, cv2.THRESH_BINARY)

    attempt = 0
    while attempt < max_attempts:
        max_x, max_y = w_bg - tag_image_binary.shape[1], h_bg - tag_image_binary.shape[0]
        if max_x <= 0 or max_y <= 0:
            return None, None, None
        x, y = random.randint(0, max_x), random.randint(0, max_y)
        bbox = (x, y, tag_image_binary.shape[1], tag_image_binary.shape[0])

        if not check_overlap(bbox, existing_bboxes):
            # Extract the region of interest (ROI) from the background
            roi = background_image[y:y + tag_image_binary.shape[0], x:x + tag_image_binary.shape[1]]

            # Create binary masks for blending
            tag_mask = (tag_image_binary < 128).astype(np.uint8) * 255  # Black regions
            tag_mask_inv = cv2.bitwise_not(tag_mask)

            # Blend the tag with the background
            img_bg = cv2.bitwise_and(roi, roi, mask=tag_mask_inv)
            tag_fg = cv2.bitwise_and(cv2.cvtColor(tag_image_binary, cv2.COLOR_GRAY2BGR), cv2.cvtColor(tag_image_binary, cv2.COLOR_GRAY2BGR), mask=tag_mask)

            # Overlay tag foreground onto the background
            dst = cv2.add(img_bg, tag_fg)
            background_image[y:y + dst.shape[0], x:x + dst.shape[1]] = dst

            # Create the full mask for the tag
            full_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
            full_mask[y:y + tag_mask.shape[0], x:x + tag_mask.shape[1]] = tag_mask

            return background_image, bbox, full_mask
        else:
            attempt += 1
    return None, None, None





# Function to create a single image in the dataset with validation
def create_single_image(args):
    image_count, tag_ids, config = args
    try:
        # Load a random background image
        bg_image = load_random_background()
        if bg_image is None:
            return  # Skip this iteration if no background is loaded

        h_bg, w_bg = bg_image.shape[:2]
        mask_image = np.zeros((h_bg, w_bg), dtype=np.uint8)

        # Determine the number of tags for this image
        num_tags_in_image = random.randint(config['num_tags_per_image'][0], config['num_tags_per_image'][1])
        existing_bboxes = []
        valid_tags_added = 0

        for _ in range(num_tags_in_image):
            tag_id = random.choice(tag_ids)

            # Set a random tag size between 100 and 400
            random_tag_size = random.randint(100, 400)
            scale_factor = random.uniform(config['scale_factor_range'][0], config['scale_factor_range'][1])

            # Generate the AprilTag with the random size
            tag_image = generate_apriltag(tag_id, config['tag_family'], random_tag_size)

            if tag_image is None or tag_image.size == 0:
                continue

            # Warp, rotate, and overlay the tag onto the background
            result = overlay_apriltag(bg_image, tag_image, existing_bboxes, scale_factor=scale_factor)
            if result is None or result[0] is None:
                continue

            bg_image, bbox, tag_mask = result
            if bg_image is None or tag_mask is None or tag_mask.size == 0:
                continue

            mask_image = cv2.bitwise_or(mask_image, tag_mask)
            existing_bboxes.append(bbox)
            valid_tags_added += 1

        # Ensure at least one tag was added
        if valid_tags_added == 0:
            return create_single_image(args)

        images_dir = os.path.join(config['output_dir'], 'images')
        masks_dir = os.path.join(config['output_dir'], 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        image_filename = os.path.join(images_dir, f"image_{image_count}.jpg")
        mask_filename = os.path.join(masks_dir, f"mask_{image_count}.png")

        if not cv2.imwrite(image_filename, bg_image):
            print(f"Failed to save image {image_filename}")
        if not cv2.imwrite(mask_filename, mask_image):
            print(f"Failed to save mask {mask_filename}")

    except Exception as e:
        print(f"Error generating image {image_count}: {e}")


# Main function for multiprocessing and validation
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
