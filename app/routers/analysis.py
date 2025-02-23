from fastapi import APIRouter, UploadFile, File
from app.services.llm_service import analyze_medical_scan
import shutil

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_path = f"static/{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_medical_scan(image_path)
    return {"analysis_result": result}
