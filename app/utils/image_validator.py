def is_medical_scan(image_data: bytes, gemini_result=None) -> bool:
    """
    Determine if an image is a medical scan using the provided Gemini analysis result.
    """
    # If Gemini result is provided, use it to validate
    if gemini_result:
        # Extract scan type from the structured result
        scan_type = gemini_result.get("scan_type", "").lower()
        # Check if recognized as a medical scan type
        return any(term in scan_type for term in ["mri", "ct", "x-ray", "scan", "medical"])

    # If no result provided, assume it's valid and let the main analysis determine
    return True

