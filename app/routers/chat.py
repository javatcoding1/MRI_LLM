from fastapi import APIRouter, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import mimetypes
import uuid

from app.utils.cache import ImageCache
from app.utils.image_validator import is_medical_scan
from app.utils.RegionOfIntrest import process_mri_image
from app.services.llm_service import analyze_medical_scan, get_medical_chat_response
from app.utils.ResponseParser import parse_medical_scan_result
from app.services.classification_service import predict_tumor_from_memory

router = APIRouter()

# Initialize cache
image_cache = ImageCache()

# Chat sessions storage
chat_sessions = {}

def get_session_id(session_id: Optional[str] = Form(None)):
    """Get or generate a session ID"""
    if not session_id:
        return str(uuid.uuid4())
    return session_id

@router.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None),
    session_id: str = Depends(get_session_id)
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
        "image_analysis": [],
        "session_id": session_id
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
                
            # Validate if it's a medical image
            if not is_medical_scan(contents):
                image_analysis_tasks.append({
                    "filename": image.filename,
                    "error": "The uploaded image does not appear to be a medical scan."
                })
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
                
                # Get LLM analysis to determine organ type
                mime_type = mimetypes.guess_type(image.filename)[0] or "image/jpeg"
                raw_result = analyze_medical_scan(contents, mime_type)
                structured_result = parse_medical_scan_result(raw_result)
                
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
                
            except Exception as e:
                image_analysis_tasks.append({
                    "filename": image.filename,
                    "error": str(e)
                })
        
        response["image_analysis"] = image_analysis_tasks
        
        # Generate a response that incorporates the analysis results
        if message:
            # Format a prompt that includes analysis results for Gemini
            analysis_summary = []
            for task in image_analysis_tasks:
                if "analysis" in task:
                    analysis = task["analysis"]
                    summary = f"Image {task['filename']}: "
                    if "llm_analysis" in analysis:
                        llm = analysis["llm_analysis"]
                        summary += f"Scan type: {llm.get('scan_type', 'Unknown')}, "
                        summary += f"Organ: {llm.get('organ', 'Unknown')}, "
                        summary += f"Tumor: {llm.get('tumor_type', 'Unknown')}"
                    if "tumor_prediction" in analysis and analysis["tumor_prediction"]:
                        pred = analysis["tumor_prediction"]
                        summary += f", Prediction: {pred.get('predicted_class', 'Unknown')} "
                        summary += f"(Confidence: {pred.get('confidence_level', 0):.2f})"
                    analysis_summary.append(summary)
            
            # Create a context-aware prompt for Gemini
            context_prompt = f"User message: {message}\n\nImage Analysis Results:\n" + "\n".join(analysis_summary)
            
            # Get response from Gemini
            chat_response = get_medical_chat_response(context_prompt, session_id)
            response["message"] = chat_response
    
    # Process text-only message if no images were provided
    elif message:
        # Use Gemini for general medical imaging questions
        chat_response = get_medical_chat_response(message, session_id)
        response["message"] = chat_response
    
    return response