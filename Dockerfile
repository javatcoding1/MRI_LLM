# Use Python 3.12.4 as base image
FROM python:3.12.4-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for OpenCV, Git, and Large File Support)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory and clone Hugging Face repo inside it
RUN mkdir -p models && \
    cd models && \
    git lfs install && \
    git clone https://huggingface.co/wizaye/MRI_LLM && \
    mv MRI_LLM/* . && \
    rm -rf MRI_LLM

# Copy project files
COPY . .

# Create .env file inside the container
RUN echo "GENAI_API_KEY=your_actual_api_key_here" > /app/.env

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
