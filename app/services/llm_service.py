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