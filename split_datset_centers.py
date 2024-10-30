import os
import glob
import random
import shutil


def split_dataset(dataset_dir='dataset', train_ratio=0.8):
    """
    Splits the dataset into training and validation sets, moving the images, labels, and localization files
    to the appropriate directories.

    Args:
        dataset_dir (str): The root directory containing the dataset.
        train_ratio (float): The ratio of data to use for training. The rest will be used for validation.
    """
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'localization', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'localization', 'val'), exist_ok=True)

    # Get list of all image files in the dataset directory
    images = glob.glob(os.path.join(dataset_dir, '*.jpg'))
    images.sort()
    random.seed(42)  # For reproducibility
    random.shuffle(images)

    # Determine the split counts
    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    def move_files(image_paths, split_type):
        """
        Moves image, label, and localization files to the appropriate directories.

        Args:
            image_paths (list): List of image paths to move.
            split_type (str): Either 'train' or 'val', to indicate which split the files belong to.
        """
        for img_path in image_paths:
            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(base_name)[0]
            label_path = os.path.join(dataset_dir, name_without_ext + '.txt')
            loc_path = os.path.join(dataset_dir, name_without_ext + '_loc.txt')

            # Move the image
            shutil.move(img_path, os.path.join(dataset_dir, 'images', split_type, base_name))

            # Move label if it exists
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(dataset_dir, 'labels', split_type, name_without_ext + '.txt'))
            else:
                print(f"Label file not found for image {img_path}")

            # Move localization file if it exists
            if os.path.exists(loc_path):
                shutil.move(loc_path,
                            os.path.join(dataset_dir, 'localization', split_type, name_without_ext + '_loc.txt'))
            else:
                print(f"Localization file not found for image {img_path}")

    # Move training images, labels, and localization files
    print("Moving training files...")
    move_files(train_images, 'train')

    # Move validation images, labels, and localization files
    print("Moving validation files...")
    move_files(val_images, 'val')

    print("Dataset split completed successfully.")


# Run the function
if __name__ == "__main__":
    split_dataset(dataset_dir='dataset_centers', train_ratio=0.8)
