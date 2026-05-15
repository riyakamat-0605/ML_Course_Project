"""
predict.py
==========
ISL Recognition — Prediction Module (Fixed)
Riya Kamat | 01FE24BCA190

Functions exported (matches Sonali's app.py exactly):
    decode_image(base64_string)         → numpy image
    extract_landmarks_from_image(img)   → numpy array (127,) or None
    predict_gesture(landmarks)          → (label, confidence)
"""

import base64
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM

# ==========================================
# FIX FOR KERAS VERSION COMPATIBILITY
# ==========================================
class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

tf.keras.utils.get_custom_objects()['LSTM'] = FixedLSTM

# ─────────────────────────────────────────────
# CONFIG — must match training exactly
# ─────────────────────────────────────────────
SEQUENCE_LENGTH  = 30
FEATURE_DIM      = 127    # 1 (two-hand flag) + 63 (left) + 63 (right zeros)
CONF_THRESHOLD   = 0.6
SMOOTHING_WINDOW = 10
MIN_VOTES        = 5

# ─────────────────────────────────────────────
# LABEL MAP — matches labels.py and training
# ─────────────────────────────────────────────
LABEL_MAP = {
    "0": "hello",
    "1": "thanks",
    "2": "yes",
}

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("[predict.py] Loading LSTM model...")
try:
    model = tf.keras.models.load_model(
        'model/isl_lstm_combined.h5',
        custom_objects={'LSTM': FixedLSTM}
    )
    print("[predict.py] Model loaded successfully!")
except Exception as e:
    raise RuntimeError(
        f"[predict.py] Could not load model.\n"
        f"Make sure model/isl_lstm_combined.h5 exists.\n"
        f"Error: {e}"
    )

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
_mp_hands = mp.solutions.hands
_hands    = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────────────
# INTERNAL BUFFERS (persist between API calls)
# ─────────────────────────────────────────────
_sequence:    list  = []
_pred_buffer: deque = deque(maxlen=SMOOTHING_WINDOW)

# ─────────────────────────────────────────────
# FUNCTION 1 — decode_image
# Called by Sonali's app.py: img = decode_image(data['image'])
# ─────────────────────────────────────────────
def decode_image(base64_string: str):
    """
    Convert base64 image string to OpenCV image (numpy array).
    Strips data URL prefix if present.
    Returns numpy BGR image or None on failure.
    """
    try:
        # Strip data URL prefix if frontend sends it
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img  # None if decode failed
    except Exception as e:
        print(f"[decode_image] Error: {e}")
        return None


# ─────────────────────────────────────────────
# FUNCTION 2 — extract_landmarks_from_image
# Called by Sonali's app.py: landmarks = extract_landmarks_from_image(img)
# Returns 127-feature vector or None if no hand detected
# ─────────────────────────────────────────────
def extract_landmarks_from_image(img):
    """
    Run MediaPipe on image and extract 127-feature vector.
    Feature format: [two_hand_flag(1), primary_hand_xyz(63), zeros(63)]
    Matches exactly what collect_sequence.py and feature_builder.py produce.
    Returns numpy array (127,) or None if no hand detected.
    """
    if img is None:
        return None

    try:
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = _hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None  # No hand detected — caller handles this

        # Extract landmarks from first hand
        hand_landmarks = result.multi_hand_landmarks[0]
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

        uses_two_hands = 1 if len(result.multi_hand_landmarks) == 2 else 0

        # Build 127-feature vector — same format as training
        feature_vector = np.array(
            [uses_two_hands] + data + [0.0] * 63,
            dtype=np.float32
        )
        return feature_vector  # shape: (127,)

    except Exception as e:
        print(f"[extract_landmarks_from_image] Error: {e}")
        return None


# ─────────────────────────────────────────────
# FUNCTION 3 — predict_gesture
# Called by Sonali's app.py: label, confidence = predict_gesture(landmarks)
# Maintains internal 30-frame buffer across API calls
# ─────────────────────────────────────────────
def predict_gesture(landmarks):
    """
    Accumulate landmarks into a 30-frame buffer.
    Run LSTM once buffer is full.
    Returns (label, confidence) — label is None until buffer is full
    and prediction is confident enough.

    Args:
        landmarks: numpy array (127,) from extract_landmarks_from_image

    Returns:
        (label: str or None, confidence: float)
    """
    global _sequence, _pred_buffer

    if landmarks is None:
        # No hand — drain stale frames
        if _sequence:
            _sequence.pop(0)
        return None, 0.0

    # Add to buffer
    _sequence.append(landmarks)
    if len(_sequence) > SEQUENCE_LENGTH:
        _sequence.pop(0)

    # Need full 30-frame sequence before predicting
    if len(_sequence) < SEQUENCE_LENGTH:
        return None, 0.0

    # Run LSTM
    input_data = np.array(_sequence, dtype=np.float32).reshape(
        1, SEQUENCE_LENGTH, FEATURE_DIM
    )

    try:
        probs      = model.predict(input_data, verbose=0)[0]
        pred       = int(np.argmax(probs))
        confidence = float(np.max(probs))
    except Exception as e:
        print(f"[predict_gesture] Model prediction error: {e}")
        return None, 0.0

    # Smoothing buffer
    _pred_buffer.append(pred)
    final_pred = max(set(_pred_buffer), key=_pred_buffer.count)
    vote_count = _pred_buffer.count(final_pred)

    # Only return label when confident AND stable
    if vote_count > MIN_VOTES and confidence > CONF_THRESHOLD:
        label = LABEL_MAP.get(str(final_pred), "Unknown")
        return label, round(confidence, 2)

    return None, round(confidence, 2)


def reset_buffers():
    """Clear sequence and prediction buffers. Call via /reset endpoint."""
    global _sequence, _pred_buffer
    _sequence    = []
    _pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
    print("[predict.py] Buffers reset.")


print("predict.py loaded successfully!")