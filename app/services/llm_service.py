import google.generativeai as genai
import base64
import mimetypes

from app.config import GENAI_API_KEY

genai.configure(api_key=GENAI_API_KEY)

def analyze_medical_scan(image_path: str):
    # Detect MIME type
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpg"

    # Read image file as binary data
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Choose Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # System prompt
    prompt = """You are analyzing a medical scan image. Provide structured output:
    - **Scan Type:** (MRI, CT Scan, X-ray)
    - **Organ:** (Brain, Lung, Heart, Breast)
    - **Tumor Type:** (Specify if detected)
    - **Tumor Subclass:** (If applicable)
    - **Detailed Description:** (Size, shape, location)
    - **Possible Causes:** (Genetic, environmental, lifestyle)
    - **Clinical Insights:** (Medical observations)
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
