from fastapi import APIRouter, UploadFile, File
from app.services.classification_service import predict_tumor
import shutil

router = APIRouter()


@router.post("/predict/{organ_type}")
async def predict_tumor_endpoint(organ_type: str, file: UploadFile = File(...)):
    image_path = f"static/{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_tumor(image_path, organ_type)
    return result
