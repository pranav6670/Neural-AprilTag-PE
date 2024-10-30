import os
import cv2
import numpy as np

def visualize_dataset(images_dir, masks_dir, overlay_dir=None, num_samples=10):
    """
    Visualize images and their corresponding masks overlaid.

    Args:
        images_dir (str): Directory containing images.
        masks_dir (str): Directory containing masks.
        overlay_dir (str, optional): Directory to save overlaid images. If None, images are not saved.
        num_samples (int): Number of samples to visualize.
    """
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    # Ensure the number of images and masks are the same
    assert len(image_files) == len(mask_files), "Number of images and masks do not match."

    # Create overlay directory if it doesn't exist and if saving is desired
    if overlay_dir is not None:
        os.makedirs(overlay_dir, exist_ok=True)

    for i in range(1, num_samples, 10):
        image_path = os.path.join(images_dir, image_files[i])
        mask_path = os.path.join(masks_dir, mask_files[i])

        # Read image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if image and mask are read correctly
        if image is None:
            print(f"Error reading image {image_path}")
            continue
        if mask is None:
            print(f"Error reading mask {mask_path}")
            continue

        # Resize mask to match image dimensions (if necessary)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create a color mask
        color_mask = np.zeros_like(image)
        color_mask[:, :, 2] = mask  # Set the red channel

        # Overlay the mask onto the image
        alpha = 0.5  # Transparency factor
        overlay = cv2.addWeighted(image, 1, color_mask, alpha, 0)

        # Display using OpenCV
        cv2.imshow('Overlay', overlay)
        cv2.waitKey(0)  # Wait for a key press to move to the next image

        # Alternatively, display using matplotlib
        # plt.figure(figsize=(10, 6))
        # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        # Save the overlaid image if desired
        if overlay_dir is not None:
            overlay_path = os.path.join(overlay_dir, f"overlay_{i}.jpg")
            cv2.imwrite(overlay_path, overlay)

    # Close any OpenCV windows
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    images_dir = '../dataset_segmentation/images'
    masks_dir = '../dataset_segmentation/masks'
    overlay_dir = './viz_overlays'  # Directory to save overlaid images (optional)

    # Visualize 20 samples from the dataset
    visualize_dataset(images_dir, masks_dir, overlay_dir=overlay_dir, num_samples=200)
