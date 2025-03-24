from app.services.llm_service import analyze_medical_scan
from app.utils.ResponseParser import parse_medical_scan_result

def is_medical_scan(image_data: bytes) -> bool:
    """
    Use Gemini to determine if an image is a medical scan.
    """
    try:
        # Detect MIME type
        mime_type = "image/jpeg"  # Default
        
        # Get preliminary analysis from Gemini
        raw_result = analyze_medical_scan(image_data, mime_type)
        structured_result = parse_medical_scan_result(raw_result)
        
        # Check if scan type is identified
        scan_type = structured_result.get("scan_type", "").lower()
        return any(term in scan_type for term in ["mri", "ct", "x-ray", "scan", "medical"])
    except:
        return False