
from fastapi import FastAPI, File, UploadFile
from app.routers import analysis, prediction
from fastapi.responses import JSONResponse
from app.utils.RegionOfIntrest import process_mri_image
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
async def process_image(file: UploadFile = File(...)):
    """
    Process uploaded MRI image and return ROI and heatmap as base64 encoded strings.
    No files are stored on the server.
    """
    # Read file contents into memory
    contents = await file.read()

    try:
        # Process the image directly from memory
        roi_base64, heatmap_base64 = process_mri_image(contents)

        response = {"heatmap": heatmap_base64,"regionofintrest": roi_base64}

        if roi_base64 is None:
            response["message"] = "No tumor detected"
            return JSONResponse(content=response, status_code=404)

        response["roi"] = roi_base64
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
