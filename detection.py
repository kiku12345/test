import os
import cv2
import numpy as np
from keras.models import load_model

# Define the path to the model file
MODEL_PATH = os.path.join('static', 'DriveAlert_Model.h5')

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

# Load the model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise Exception(f"Error loading model: {e}")


# Function to preprocess input image for drowsiness detection
def preprocess_frame(frame):
    try:
        resized_frame = cv2.resize(frame, (145, 145))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame / 255.0
        return normalized_frame
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None


# Function to detect drowsiness using the model
def detect_drowsiness(frame):
    try:
        # Preprocess the image
        preprocessed_img = preprocess_frame(frame)

        if preprocessed_img is None:
            return False

        # Use the model to detect drowsiness
        prediction = model.predict(np.array([preprocessed_img]))

        # Assuming a threshold of 0.5 for binary classification
        drowsy = prediction[0] > 0.5

        return drowsy

    except Exception as e:
        print(f"Error detecting drowsiness: {e}")
        return False
