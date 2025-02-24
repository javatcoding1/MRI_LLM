import cv2
import numpy as np
import os
import uuid


def extract_roi_and_heatmap(image_path, output_dir="./static"):
    """
    Extracts the region of interest (ROI) and generates a heatmap from a grayscale MRI image.

    Parameters:
        image_path (str): Path to the input MRI image.
        output_dir (str): Directory to save the processed images.

    Returns:
        tuple: (roi_path, heatmap_path), where:
            - roi_path (str or None): Path to the extracted tumor region, None if no tumor detected.
            - heatmap_path (str): Path to the heatmap image.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Load the grayscale MRI image
    mri_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mri_image is None:
        raise ValueError("Error: Image not found or unable to load.")

    # Apply the JET colormap
    jet_colored = cv2.applyColorMap(mri_image, cv2.COLORMAP_JET)

    # Convert to HSV to isolate the orange and red colors
    hsv_image = cv2.cvtColor(jet_colored, cv2.COLOR_BGR2HSV)

    # Define color ranges for tumor detection
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the tumor regions
    mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    combined_mask = cv2.bitwise_or(mask_orange, mask_red)

    # Apply morphological operations for noise reduction
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours to detect the largest region
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    roi_path = None
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        roi = mri_image[y:y + h, x:x + w]

        # Save ROI as an image
        roi_filename = f"roi_{uuid.uuid4().hex}.png"
        roi_path = os.path.join(output_dir, roi_filename)
        cv2.imwrite(roi_path, roi)

    # Save heatmap image
    heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    cv2.imwrite(heatmap_path, jet_colored)

    return roi_path, heatmap_path
