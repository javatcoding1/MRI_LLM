import io
import os

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from app.routers import analysis, prediction
import base64
from fastapi.responses import JSONResponse
from app.utils.RegionOfIntrest import extract_roi_and_heatmap
app = FastAPI(title="Medical Scan Analysis API")

# Include API endpoints
app.include_router(analysis.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to Medical Scan Analysis API"}


# Try 1 : The User uploads a photo and the photo should be temporarily stored in static folder
# Try 2 : Try Using Blob or a array.
@app.post("/process-image")
async def process_image(file: UploadFile = File(...), STATIC_DIR="./static"):
    temp_image_path = os.path.join(STATIC_DIR, file.filename)

    # Save uploaded image temporarily
    with open(temp_image_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process the image
    roi_path, heatmap_path = extract_roi_and_heatmap(temp_image_path)

    # Remove the uploaded image after processing
    os.remove(temp_image_path)

    if roi_path is None:
        return JSONResponse(content={"error": "No tumor detected"}, status_code=404)

    return JSONResponse(content={"roi_path": roi_path, "heatmap_path": heatmap_path})
