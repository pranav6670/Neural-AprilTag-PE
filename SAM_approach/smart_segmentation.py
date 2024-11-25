import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

def detect_tags_with_black_borders(image_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate masks automatically
    masks = mask_generator.generate(image)

    # Debug: Visualize all generated masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image[..., ::-1])
    for mask in masks:
        plt.contour(mask['segmentation'], colors=['blue'], levels=[0.5])
    plt.title('All Generated Masks')
    plt.axis('off')
    plt.show()

    # Filter masks to find ones with black borders
    selected_mask = None
    for mask in masks:
        # Convert the mask to uint8
        mask_region = (mask['segmentation'] * 255).astype(np.uint8)

        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask_region)

        # Convert to grayscale for border analysis
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Find contours of the mask
        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Get the largest contour (assume it's the relevant tag)
        contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 sides (quadrilateral)
        if len(approx) != 4:
            continue

        # Extract the bounding box for border analysis
        x, y, w, h = cv2.boundingRect(contour)
        border = gray[y:y+h, x:x+w]

        # Check if the border is predominantly black
        border_pixels = border.flatten()
        black_pixel_ratio = np.sum(border_pixels < 50) / len(border_pixels)  # Adjust threshold as needed

        if black_pixel_ratio > 0.6:  # Adjust threshold based on your dataset
            selected_mask = mask
            break

    if selected_mask is None:
        print("No suitable tag with a black border was found.")
        return

    # Visualize the selected mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image[..., ::-1])
    plt.imshow(selected_mask['segmentation'], alpha=0.5, cmap='jet')
    plt.title('Selected Mask with Black Border')
    plt.axis('off')
    plt.show()

    # Extract the selected mask and refine it
    mask = selected_mask['segmentation'].astype(np.uint8) * 255

    # Find contours again for the selected mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contours found in the mask.")
        return

    # Assume the largest contour is the tag
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Debug: Draw the contour approximation
    contour_image = image.copy()
    cv2.drawContours(contour_image, [approx_polygon], -1, (0, 255, 0), 2)
    cv2.imshow('Contour Approximation', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Check if the polygon has 4 sides
    if len(approx_polygon) == 4:
        corners = approx_polygon.reshape(4, 2)
        print("Detected a quadrilateral for the tag.")
    else:
        print("No quadrilateral found in the segmentation.")
        return

    # Visualize the detected corners
    for point in corners:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    cv2.imshow('Detected Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the corners for further processing
    return corners


image_path = r"C:\Users\prana\AprilTags\CapturedImages\image_000463_Yaw0.0_Pitch30.0_Roll70.0.png"
detect_tags_with_black_borders(image_path)
