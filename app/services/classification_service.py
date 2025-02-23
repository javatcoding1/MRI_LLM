import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from app.config import BRAIN_MODEL_PATH, LUNG_MODEL_PATH, BREAST_MODEL_PATH

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path, img_size):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_tumor(img_path, organ_type):
    if organ_type == "Brain":
        model = load_model(BRAIN_MODEL_PATH)
        img_size = (299, 299)
        class_labels = ["Glioma", "Meningioma", "Pituitary Tumor", "Normal"]
    elif organ_type == "Lung":
        model = load_model(LUNG_MODEL_PATH)
        img_size = (224, 224)
        class_labels = ["Benign", "Malignant", "Normal"]
    else:
        model = load_model(BREAST_MODEL_PATH)
        img_size = (244, 244)
        class_labels = ["Benign", "Malignant"]

    img_array = preprocess_image(img_path, img_size)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return {
        "predicted_class": predicted_class,
        "confidence_scores": prediction.tolist()
    }
