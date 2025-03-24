import google.generativeai as genai
import base64
from app.config import GENAI_API_KEY

genai.configure(api_key=GENAI_API_KEY)

def analyze_medical_scan(image_data: bytes, mime_type: str):
    import google.generativeai as genai
    from app.config import GENAI_API_KEY

    genai.configure(api_key=GENAI_API_KEY)

    # Choose Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Modified prompt to encourage a more structured response
    prompt = """You are analyzing a medical scan image. Provide structured output with EXACTLY these fields:
    - Scan Type (MRI, CT Scan, X-ray)
    - Organ (Brain, Lung, Heart, Breast)
    - Tumor Type (Specify if detected)
    - Tumor Subclass (If applicable)
    - Detailed Description (Size, shape, location)
    - Possible Causes (Genetic, environmental, lifestyle)
    - Clinical Insights (Medical observations)

    Format your response using these EXACT field names with a colon after each field name.
    """

    # Generate response
    response = model.generate_content(
        [
            {
                "mime_type": mime_type,
                "data": base64.b64encode(image_data).decode("utf-8"),
            },
            prompt,
        ]
    )

    return response.text


# Add this function to your existing llm_service.py file

def get_medical_chat_response(message: str, session_id: str = None):
    """
    Get a response from Gemini for medical imaging related questions.
    Maintains conversation context using session_id.
    """
    # Choose Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # System prompt to guide the model's behavior
    system_prompt = """You are a medical imaging assistant specializing in MRI, CT scans, and other medical imaging technologies.
    Your primary focus is helping users understand medical scans, tumor detection, and related medical concepts.
    
    Guidelines:
    1. Provide accurate, helpful information about medical imaging, tumors, and scan interpretation.
    2. If asked about your identity, say you are a medical imaging assistant without mentioning Gemini.
    3. Do not respond to personal questions or topics unrelated to medical imaging.
    4. When discussing scan results, emphasize that these are computational analyses and not medical diagnoses.
    5. Always recommend consulting healthcare professionals for actual medical advice.
    
    Please respond to the user's query about medical imaging:
    """

    # Combine system prompt with user message
    full_prompt = f"{system_prompt}\n\nUser: {message}"

    # Generate response
    response = model.generate_content(full_prompt)

    return response.text