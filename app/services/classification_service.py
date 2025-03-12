import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image


def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image_from_memory(img_data, img_size):
    """
    Process image data from memory into the format expected by the model.
    Ensures correct dimensions (height, width, channels).
    """
    # Open image from binary data using PIL
    img = Image.open(BytesIO(img_data))

    # Convert to RGB mode to ensure 3 channels
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image to expected dimensions
    img = img.resize(img_size)

    # Convert to numpy array with correct shape (height, width, channels)
    img_array = np.array(img)

    # Ensure we have a 4D tensor (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0

    return img_array


def predict_tumor_from_memory(img_data, organ_type):
    # Import here to avoid circular imports
    from app.config import BRAIN_MODEL_PATH, LUNG_MODEL_PATH, BREAST_MODEL_PATH

    if organ_type == "Brain":
        model = load_model(BRAIN_MODEL_PATH)
        img_size = (299, 299)
        class_labels = ["Glioma", "Meningioma", "Pituitary Tumor", "Normal"]
    elif organ_type == "Lung":
        model = load_model(LUNG_MODEL_PATH)
        img_size = (224, 224)
        class_labels = ["Benign", "Malignant", "Normal"]
    else:  # Breast
        model = load_model(BREAST_MODEL_PATH)
        img_size = (244, 244)
        class_labels = ["Benign", "Malignant"]

    try:
        # Process image directly from memory
        img_array = preprocess_image_from_memory(img_data, img_size)

        # Print shape for debugging
        print(f"Processed image shape: {img_array.shape}")

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        # Check if the highest probability is below a threshold
        confidence = float(np.max(prediction))  # Convert to Python float
        confidence_threshold = 0.5  # You can adjust this threshold

        if confidence < confidence_threshold:
            prediction_status = "Low Confidence"
        else:
            prediction_status = "High Confidence"

        return {
            "predicted_class": predicted_class,
            "confidence_scores": prediction.tolist()[0],  # Convert to list and extract from batch
            "confidence_level": confidence,
            "prediction_status": prediction_status
        }
    except Exception as e:
        # Add more diagnostic information to the error
        error_message = f"Error during prediction: {str(e)}"
        if 'img_array' in locals():
            error_message += f"\nImage array shape: {img_array.shape}"
        raise Exception(error_message)