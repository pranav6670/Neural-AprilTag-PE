import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_sample_with_corners(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    h, w, _ = image.shape  # Get image dimensions

    # Load the labels
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # Process each label entry
    for label in labels:
        values = list(map(float, label.strip().split()))
        if len(values) != 11:
            print(f"Incorrect label format in {label_path}")
            continue

        # Extract label values
        class_id = int(values[0])
        x_center_norm, y_center_norm = values[1], values[2]
        x_tl_norm, y_tl_norm = values[3], values[4]
        x_tr_norm, y_tr_norm = values[5], values[6]
        x_br_norm, y_br_norm = values[7], values[8]
        x_bl_norm, y_bl_norm = values[9], values[10]

        # Convert normalized coordinates to pixel values
        x_center = int(x_center_norm * w)
        y_center = int(y_center_norm * h)
        x_tl = int(x_tl_norm * w)
        y_tl = int(y_tl_norm * h)
        x_tr = int(x_tr_norm * w)
        y_tr = int(y_tr_norm * h)
        x_br = int(x_br_norm * w)
        y_br = int(y_br_norm * h)
        x_bl = int(x_bl_norm * w)
        y_bl = int(y_bl_norm * h)

        # Draw the corners as red dots
        corners = [(x_tl, y_tl), (x_tr, y_tr), (x_br, y_br), (x_bl, y_bl)]
        for (x, y) in corners:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red dots for corners

        # Draw the center as a blue dot
        cv2.circle(image, (x_center, y_center), 5, (255, 0, 0), -1)  # Blue dot for center

        # Draw green lines connecting the corners
        pts = np.array(corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # Green polygon

        # Display the Class ID
        cv2.putText(image, f"ID: {class_id}", (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the image with matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axes
    plt.show()


# Example usage:
visualize_sample_with_corners('dataset_centers/images/train/image_0.jpg',
                              'dataset_centers/labels/train/image_0.txt')
