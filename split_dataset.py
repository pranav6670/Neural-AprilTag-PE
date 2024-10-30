import os
import glob
import random
import shutil

def split_dataset(dataset_dir='dataset', train_ratio=0.8):
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)

    # Get list of all image files in the dataset directory
    images = glob.glob(os.path.join(dataset_dir, '*.jpg'))
    images.sort()
    random.seed(42)  # For reproducibility
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    # Move training images and labels
    for img_path in train_images:
        base_name = os.path.basename(img_path)
        label_path = os.path.splitext(img_path)[0] + '.txt'

        # Move image
        shutil.move(img_path, os.path.join(dataset_dir, 'images', 'train', base_name))

        # Move label if it exists
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(dataset_dir, 'labels', 'train', base_name.replace('.jpg', '.txt')))
        else:
            print(f"Label file not found for image {img_path}")

    # Move validation images and labels
    for img_path in val_images:
        base_name = os.path.basename(img_path)
        label_path = os.path.splitext(img_path)[0] + '.txt'

        # Move image
        shutil.move(img_path, os.path.join(dataset_dir, 'images', 'val', base_name))

        # Move label if it exists
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(dataset_dir, 'labels', 'val', base_name.replace('.jpg', '.txt')))
        else:
            print(f"Label file not found for image {img_path}")

    print("Dataset split completed successfully.")

# Run the function
split_dataset()
