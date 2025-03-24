from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.utils.RegionOfIntrest import process_mri_image

router = APIRouter()

@router.post("/process-image")
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

        response = {"heatmap": heatmap_base64, "regionofintrest": roi_base64}

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