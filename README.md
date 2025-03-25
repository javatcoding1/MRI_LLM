# ScanSage AI - Backend

The **ScanSage AI Backend** is a FastAPI-based server that powers the medical imaging analysis platform. It provides endpoints for processing medical scans, detecting tumors, generating heatmaps, and responding to natural language queries using AI.

## Features

- **FastAPI-powered REST API** for efficient request handling.
- **Medical Scan Processing** for MRI, CT, and X-ray images.
- **Asynchronous Image Analysis** for parallel processing.
- **Tumor Detection & ROI Extraction** for brain, lung, and breast scans.
- **Heatmap Generation** to visualize areas of interest.
- **AI-driven Natural Language Processing** for contextual medical insights.
- **SHA-256 Caching System** to optimize redundant image processing.
- **Multi-Modal Analysis** combining image and text-based queries.

## Backend Architecture

The backend is designed for high-performance medical image analysis and AI-powered insights.

- **Framework**: FastAPI (Python 3.8+)
- **AI Models**:
  - Custom tumor classification models (Brain, Lung, Breast)
  - Google's Gemini 1.5 Pro for NLP and image analysis
- **Image Processing**: OpenCV & NumPy
- **Caching**: SHA-256 based in-memory caching
- **Concurrency**: Async processing with ThreadPoolExecutor

## Setup Guide

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) Docker for containerized deployment

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MPLLM
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root:
   ```
   GENAI_API_KEY=your_google_genai_api_key
   ```

5. Ensure model files are available in the `models/` directory:
   ```
   models/
   ├── brain_model.h5
   ├── lung_tumor.h5
   ├── breast_tumor.h5
   ```

### Running the Backend

#### With Uvicorn (Development Mode)
```bash
uvicorn app.main:app --reload
```
Access the API at `http://localhost:8000`.

#### With Docker (Production Mode)
1. Build the Docker image:
   ```bash
   docker build -t scansage-backend .
   ```
2. Run the container:
   ```bash
   docker run -d -p 8000:8000 --name scansage-container scansage-backend
   ```
3. API available at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Upload and analyze a medical scan. |
| `/api/chat` | POST | Submit text queries with optional medical images. |
| `/process-image` | POST | Process MRI images to extract ROI and heatmaps. |

### Example API Usage

#### Analyze a Medical Scan
```python
import requests

url = "http://localhost:8000/api/analyze"
files = {"file": open("scan.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

#### Chat with Context and Image
```python
import requests

url = "http://localhost:8000/api/chat"
files = {"images": open("scan.jpg", "rb")}
data = {"message": "Is there a tumor visible?"}
response = requests.post(url, files=files, data=data)
print(response.json())
```

## Future Enhancements

- Support for DICOM medical imaging format.
- Enhanced multi-organ classification models.
- Improved AI explainability and insights.
- Web-based UI for non-technical users.
- Longitudinal analysis for tracking tumor progression.

## Contributing

We welcome contributions! Please check our [contribution guidelines](CONTRIBUTING.md) for details on how to contribute.

## Disclaimer

This backend is for **research and educational purposes only**. It is not a substitute for clinical diagnosis or medical advice.

## Acknowledgements

- Open-source libraries including FastAPI, OpenCV, and TensorFlow and Huggingface LLMs.

