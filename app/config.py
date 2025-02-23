import os
from dotenv import load_dotenv

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
BRAIN_MODEL_PATH = "models/brain_model.h5"
LUNG_MODEL_PATH = "models/lung_tumor.h5"
BREAST_MODEL_PATH = "models/breast_tumor.h5"
