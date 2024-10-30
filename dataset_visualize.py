import cv2
import matplotlib.pyplot as plt

def visualize_sample(image_path, label_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.strip().split())
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage:
visualize_sample('dataset/images/train/image_0.jpg', 'dataset/labels/train/image_0.txt')
