import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

def analyze_masks(image_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)

    # Generate masks automatically
    masks = mask_generator.generate(image)

    # Analyze each mask
    thresholds_results = []
    for idx, mask in enumerate(masks):
        mask_region = (mask['segmentation'] * 255).astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask_region)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Calculate pixel intensity ratios
        total_pixels = gray.size
        black_pixels = np.sum(gray < 50)  # Black threshold
        white_pixels = np.sum(gray > 200)  # White threshold

        black_ratio = black_pixels / total_pixels
        white_ratio = white_pixels / total_pixels

        # Store results for this mask
        thresholds_results.append((idx, black_ratio, white_ratio))

        # Visualize the mask and pixel intensity analysis
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image[..., ::-1])
        plt.imshow(mask['segmentation'], alpha=0.5, cmap='jet')
        plt.title(f'Mask {idx}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.hist(gray.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        plt.axvline(50, color='blue', linestyle='--', label='Black Threshold (50)')
        plt.axvline(200, color='red', linestyle='--', label='White Threshold (200)')
        plt.title(f'Pixel Intensity Histogram\nMask {idx}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Mask {idx}: Black Ratio = {black_ratio:.2f}, White Ratio = {white_ratio:.2f}")

    # Print summary of thresholds
    print("\nSummary of Pixel Ratios for All Masks:")
    for idx, black_ratio, white_ratio in thresholds_results:
        print(f"Mask {idx}: Black Ratio = {black_ratio:.2f}, White Ratio = {white_ratio:.2f}")

    return thresholds_results


# Call the function
image_path = r"C:\Users\prana\AprilTags\CapturedImages\image_000463_Yaw0.0_Pitch30.0_Roll70.0.png"
thresholds = analyze_masks(image_path)
