import cv2
import numpy as np
import base64

def process_mri_image(image_data):
    """
    Processes an MRI image to detect tumors and generate a heatmap.
    Returns the ROI and heatmap as encoded images.

    Parameters:
        image_data (bytes): Raw image data.

    Returns:
        tuple: (roi_base64, heatmap_base64), where:
            - roi_base64 (str or None): Base64 encoded ROI image, None if no tumor detected.
            - heatmap_base64 (str): Base64 encoded heatmap image.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode the image
    mri_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
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

    roi_base64 = None
    if max_contour is not None and max_area > 50:  # Minimum area threshold to avoid noise
        x, y, w, h = cv2.boundingRect(max_contour)
        roi = mri_image[y:y + h, x:x + w]

        # Encode ROI as base64
        _, buffer = cv2.imencode('.png', roi)
        roi_base64 = base64.b64encode(buffer).decode('utf-8')

    # Encode heatmap as base64
    _, buffer = cv2.imencode('.png', jet_colored)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    return roi_base64, heatmap_base64