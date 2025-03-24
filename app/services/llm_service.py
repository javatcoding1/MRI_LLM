import google.generativeai as genai
import base64
from app.config import GENAI_API_KEY

genai.configure(api_key=GENAI_API_KEY)


def analyze_medical_scan_with_context(image_data: bytes = None, mime_type: str = None, message: str = None):
    """
    Unified function that analyzes a medical scan and optionally incorporates user message context.
    This eliminates redundant API calls by combining analysis and chat in one request.
    """
    # Choose Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Base analysis prompt
    base_prompt = """You are analyzing a medical scan image. Provide structured output with EXACTLY these fields:
    - Scan Type (MRI, CT Scan, X-ray)
    - Organ (Brain, Lung, Heart, Breast)
    - Tumor Type (Specify if detected)
    - Tumor Subclass (If applicable)
    - Detailed Description (Size, shape, location)
    - Possible Causes (Genetic, environmental, lifestyle)
    - Clinical Insights (Medical observations)
    Format your response using these EXACT field names with a colon after each field name.
    """

    # If a message is provided, add it to the prompt for context
    if message:
        full_prompt = f"{base_prompt}\n\nAdditionally, the user has asked: {message}\n\nFirst provide the structured analysis, then answer their question."
    else:
        full_prompt = base_prompt

    # Handle text-only queries
    if image_data is None:
        system_prompt = """You are a medical imaging assistant specializing in MRI, CT scans, and other medical imaging technologies.
        Your primary focus is helping users understand medical scans, tumor detection, and related medical concepts.
        Guidelines:
        1. Provide accurate, helpful information about medical imaging, tumors, and scan interpretation.
        2. If asked about your identity, say you are a medical imaging assistant without mentioning Gemini.
        3. Do not respond to personal questions or topics unrelated to medical imaging.
        4. When discussing scan results, emphasize that these are computational analyses and not medical diagnoses.
        5. Always recommend consulting healthcare professionals for actual medical advice.
        """
        full_prompt = f"{system_prompt}\n\nUser: {message}"
        response = model.generate_content(full_prompt)
        return response.text

    # Generate response with image
    response = model.generate_content(
        [{
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode("utf-8"),
        },
        full_prompt]
    )

    return response.text
