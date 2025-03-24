from fastapi import FastAPI
from app.routers import analysis, prediction, chat, image_processing

app = FastAPI(title="Medical Scan Analysis API")

# Include API endpoints
app.include_router(analysis.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(image_processing.router, prefix="")

@app.get("/")
async def root():
    return {"message": "Welcome to Medical Scan Analysis API"}