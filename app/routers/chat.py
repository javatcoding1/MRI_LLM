from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import mimetypes
from app.utils.cache import ImageCache
from app.utils.image_validator import is_medical_scan
from app.utils.RegionOfIntrest import process_mri_image
from app.services.llm_service import analyze_medical_scan_with_context
from app.utils.ResponseParser import parse_medical_scan_result
from app.services.classification_service import predict_tumor_from_memory

router = APIRouter()

# Initialize cache
image_cache = ImageCache()

@router.post("/chat")
async def chat_endpoint(
        message: str = Form(...),
        images: Optional[List[UploadFile]] = File(None)
):
    """
    Unified endpoint that handles both general medical imaging questions and image analysis.
    - If only text is provided, it responds with medical imaging knowledge
    - If images are provided, they're analyzed and results are returned
    - If both are provided, the images are analyzed in the context of the message
    """
    # Initialize response structure
    response = {
        "message": "",
        "image_analysis": []
    }

    # Process images if provided
    if images and len(images) > 0:
        image_analysis_tasks = []

        for image in images:
            # Read file contents into memory
            contents = await image.read()

            # Skip empty files
            if not contents:
                continue

            # Check cache first
            cached_result = image_cache.get(contents)
            if cached_result:
                image_analysis_tasks.append({
                    "filename": image.filename,
                    "analysis": cached_result,
                    "source": "cache"
                })
                continue

            # Process the image
            try:
                # Process ROI and heatmap
                roi_base64, heatmap_base64 = process_mri_image(contents)

                # Get LLM analysis and incorporate user message in a single API call
                mime_type = mimetypes.guess_type(image.filename)[0] or "image/jpeg"
                raw_result = analyze_medical_scan_with_context(contents, mime_type, message)
                structured_result = parse_medical_scan_result(raw_result)

                # Validate if it's a medical scan using the Gemini result
                if not is_medical_scan(contents, structured_result):
                    image_analysis_tasks.append({
                        "filename": image.filename,
                        "error": "The uploaded image does not appear to be a medical scan."
                    })
                    continue

                # Extract organ type for classification
                organ_type = structured_result.get("organ", "").strip()

                # Default to Brain if organ not clearly identified
                if not organ_type or organ_type.lower() not in ["brain", "lung", "breast"]:
                    if "brain" in message.lower():
                        organ_type = "Brain"
                    elif "lung" in message.lower():
                        organ_type = "Lung"
                    elif "breast" in message.lower():
                        organ_type = "Breast"
                    else:
                        organ_type = "Brain"  # Default

                # Get tumor prediction if organ is supported
                prediction_result = None
                if organ_type in ["Brain", "Lung", "Breast"]:
                    prediction_result = predict_tumor_from_memory(contents, organ_type)

                # Combine results
                analysis_result = {
                    "llm_analysis": structured_result,
                    "tumor_prediction": prediction_result,
                    "heatmap": heatmap_base64,
                    "roi": roi_base64
                }

                # Cache the result
                image_cache.set(contents, analysis_result)

                image_analysis_tasks.append({
                    "filename": image.filename,
                    "analysis": analysis_result,
                    "source": "processed"
                })

                # Extract the response message from the LLM result if it contains an answer to the user's question
                if message and "llm_response" in structured_result:
                    response["message"] = structured_result["llm_response"]

            except Exception as e:
                image_analysis_tasks.append({
                    "filename": image.filename,
                    "error": str(e)
                })

        response["image_analysis"] = image_analysis_tasks

    # Process text-only message if no images were provided or no response was extracted from image analysis
    if message and not response["message"]:
        # Use Gemini for general medical imaging questions
        response["message"] = analyze_medical_scan_with_context(None, None, message)

    return response
