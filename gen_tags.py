import cv2
import numpy as np
import os
import random
from tqdm import tqdm

def generate_apriltag(tag_id, tag_family, tag_size):
    # Get the dictionary for the specified tag family
    if tag_family == 'tag36h11':
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    elif tag_family == 'tag25h9':
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    elif tag_family == 'tag16h5':
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    else:
        print("Unknown tag family")
        return None

    # Generate the marker
    marker_image = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size)
    return marker_image

def create_random_background(width, height):
    # Generate a random faint color
    color = np.random.randint(200, 256, size=(3,), dtype=np.uint8)
    background = np.full((height, width, 3), color, dtype=np.uint8)
    return background

def check_overlap(bbox, existing_bboxes):
    x1, y1, w1, h1 = bbox
    box1 = [x1, y1, x1 + w1, y1 + h1]
    for ex_bbox in existing_bboxes:
        x2, y2, w2, h2 = ex_bbox
        box2 = [x2, y2, x2 + w2, y2 + h2]
        # Check for overlap
        if (box1[0] < box2[2] and box1[2] > box2[0] and
            box1[1] < box2[3] and box1[3] > box2[1]):
            return True  # Overlaps
    return False  # No overlap

def overlay_apriltag(background_image, tag_image, existing_bboxes, scale_factor=1.0, max_attempts=50):
    h_bg, w_bg, _ = background_image.shape
    h_tag, w_tag = tag_image.shape

    # Resize the tag image based on the scale factor
    new_w_tag = int(w_tag * scale_factor)
    new_h_tag = int(h_tag * scale_factor)
    tag_image_resized = cv2.resize(tag_image, (new_w_tag, new_h_tag))

    attempt = 0
    while attempt < max_attempts:
        # Random position where the tag will be placed
        max_x = w_bg - new_w_tag
        max_y = h_bg - new_h_tag

        if max_x <= 0 or max_y <= 0:
            print("Tag image is larger than background image.")
            return None, None

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        bbox = (x, y, new_w_tag, new_h_tag)

        # Check for overlap
        if not check_overlap(bbox, existing_bboxes):
            # No overlap, proceed to overlay
            # Convert tag image to BGR if it's grayscale
            if len(tag_image_resized.shape) == 2:
                tag_image_resized = cv2.cvtColor(tag_image_resized, cv2.COLOR_GRAY2BGR)

            # Overlay the tag onto the background image
            background_image[y:y+new_h_tag, x:x+new_w_tag] = tag_image_resized

            return background_image, bbox
        else:
            attempt += 1

    # If unable to place the tag without overlap after max_attempts
    print("Failed to place tag without overlap after maximum attempts.")
    return None, None

def generate_yolo_label(bbox, image_width, image_height, class_id=0):
    x, y, w, h = bbox
    # Convert to YOLO format: class_id, x_center, y_center, width, height (normalized)
    x_center = x + w / 2
    y_center = y + h / 2
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    return f"{class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}"

def create_dataset(num_images, tag_ids, image_width=640, image_height=480, tag_family='tag36h11', tag_size=100, output_dir='dataset'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_count = 0
    total_tags = len(tag_ids)

    print("Generating images...")
    with tqdm(total=num_images) as pbar:
        while image_count < num_images:
            # Create a random faint background image
            bg_image = create_random_background(image_width, image_height)

            num_tags_in_image = random.randint(2, 3)  # 2 to 3 tags per image
            labels = []
            existing_bboxes = []

            tags_placed = 0
            attempts = 0
            max_total_attempts = num_tags_in_image * 50  # Limit total attempts to place tags

            while tags_placed < num_tags_in_image and attempts < max_total_attempts:
                tag_id = random.choice(tag_ids)
                scale_factor = random.uniform(0.5, 1.5)  # Random scale factor
                tag_image = generate_apriltag(tag_id, tag_family, tag_size)
                if tag_image is None:
                    attempts += 1
                    continue

                # Overlay the AprilTag onto the background image without overlap
                result = overlay_apriltag(bg_image, tag_image, existing_bboxes, scale_factor=scale_factor)
                if result is None:
                    attempts += 1
                    continue
                bg_image, bbox = result

                # Add the bounding box to the list of existing boxes
                existing_bboxes.append(bbox)

                # Generate YOLO label
                label = generate_yolo_label(bbox, image_width, image_height, class_id=0)
                labels.append(label)

                tags_placed += 1

            if tags_placed == num_tags_in_image:
                # Save the image and labels
                image_filename = os.path.join(output_dir, f"image_{image_count}.jpg")
                label_filename = os.path.join(output_dir, f"image_{image_count}.txt")

                cv2.imwrite(image_filename, bg_image)
                with open(label_filename, 'w') as f:
                    f.write('\n'.join(labels))

                image_count += 1
                pbar.update(1)
            else:
                print(f"Could not place required number of tags in image {image_count}. Retrying...")
                # Proceed to the next image without incrementing image_count

    print(f"Dataset creation complete. Generated {image_count} images.")

# Example usage
# Generate a list of tag IDs (e.g., 0 to 99)
tag_ids = list(range(0, 100))

# Number of images to generate
num_images = 3000

# Create the dataset
create_dataset(num_images, tag_ids)
