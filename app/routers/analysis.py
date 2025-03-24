from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.llm_service import analyze_medical_scan_with_context
import mimetypes
from app.utils.ResponseParser import parse_medical_scan_result

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read file contents into memory
    contents = await file.read()

    try:
        # Detect MIME type from the filename
        mime_type = mimetypes.guess_type(file.filename)[0] or "image/jpeg"

        # Process the image directly from memory
        raw_result = analyze_medical_scan_with_context(contents, mime_type)

        # Convert the raw markdown-formatted result into structured JSON
        structured_result = parse_medical_scan_result(raw_result)

        return {"analysis_result": structured_result}
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
