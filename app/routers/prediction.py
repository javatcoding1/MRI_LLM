from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.classification_service import predict_tumor_from_memory

router = APIRouter()


@router.post("/predict/{organ_type}")
async def predict_tumor_endpoint(organ_type: str, file: UploadFile = File(...)):
    try:
        # Read file into memory
        contents = await file.read()

        # Process image and make prediction directly from memory
        result = predict_tumor_from_memory(contents, organ_type)

        return result
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
