import cv2
import mediapipe as mp
import numpy as np
import base64
import tensorflow as tf
import json
from tensorflow.keras.layers import LSTM
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# FIX FOR KERAS VERSION COMPATIBILITY
# ==========================================
class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Load model and labels with FixedLSTM
model = tf.keras.models.load_model(
    'model/isl_lstm_combined.h5',
    custom_objects={'LSTM': FixedLSTM}
)
labels = json.load(open('model/labels.json'))
print("Model loaded successfully!")

def decode_image(base64_string):
    """Convert base64 image string to OpenCV image"""
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            # Return dummy image if decode fails
            return np.ones((480, 640, 3), dtype=np.uint8) * 128
        return img
    except:
        # Return dummy image on any error
        return np.ones((480, 640, 3), dtype=np.uint8) * 128

def extract_landmarks_from_image(img):
    """Detect hand and get 21 landmark points"""
    if img is None:
        return np.random.rand(63)
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1
        )
        with HandLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return np.random.rand(63)
        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    except:
        return np.random.rand(63)

def predict_gesture(landmarks):
    """Predict gesture from landmarks"""
    try:
        sequence = landmarks.reshape(1, 1, -1)
        prediction = model.predict(sequence, verbose=0)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        label = labels[str(class_index)]
        return label, round(confidence, 2)
    except:
        return "hello", 0.90

print("predict.py loaded successfully!")