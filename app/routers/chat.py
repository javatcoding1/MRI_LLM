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
from asyncio import gather

router = APIRouter()

# Initialize cache
image_cache = ImageCache()

@router.post("/chat")
async def chat_endpoint(
        message: str = Form(...),
        images: Optional[List[UploadFile]] = File(None)
):
    response = {
        "message": "",
        "image_analysis": []
    }

    async def process_single_image(image):
        contents = await image.read()
        if not contents:
            return None
            
        # Check cache first
        cached_result = image_cache.get(contents)
        if cached_result:
            return {
                "filename": image.filename,
                "analysis": cached_result,
                "source": "cache"
            }

        try:
            # Batch process ROI and heatmap
            roi_base64, heatmap_base64 = process_mri_image(contents)
            
            # Batch LLM analysis
            mime_type = mimetypes.guess_type(image.filename)[0] or "image/jpeg"
            raw_results = analyze_medical_scan_with_context(contents, mime_type, message)
            structured_result = parse_medical_scan_result(raw_results)
            
            if not is_medical_scan(contents, structured_result):
                return {
                    "filename": image.filename,
                    "error": "Not a medical scan"
                }

            # Get organ type and prediction
            organ_type = structured_result.get("organ", "").strip()
            if not organ_type or organ_type.lower() not in ["brain", "lung", "breast"]:
                organ_type = "Brain"  # Default

            prediction_result = None
            if organ_type in ["Brain", "Lung", "Breast"]:
                prediction_result = predict_tumor_from_memory(contents, organ_type)

            analysis_result = {
                "llm_analysis": structured_result,
                "tumor_prediction": prediction_result,
                "heatmap": heatmap_base64,
                "roi": roi_base64
            }

            # Cache the result
            image_cache.set(contents, analysis_result)

            return {
                "filename": image.filename,
                "analysis": analysis_result,
                "source": "processed"
            }

        except Exception as e:
            return {
                "filename": image.filename,
                "error": str(e)
            }

    if images and len(images) > 0:
        # Process all images concurrently
        tasks = [process_single_image(image) for image in images]
        results = await gather(*tasks)
        response["image_analysis"] = [r for r in results if r is not None]

    # Handle text-only queries
    if message and not response["message"]:
        response["message"] = analyze_medical_scan_with_context(None, None, message)

    return response
