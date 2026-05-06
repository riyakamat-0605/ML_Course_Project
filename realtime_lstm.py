import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from feature_builder import build_feature_vector
from labels import label_map

# -------------------------
# CONFIG
# -------------------------
SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.6
SMOOTHING_WINDOW = 10
MIN_VOTES = 5

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model("models/isl_lstm_combined.h5")
inv_label_map = {v: k for k, v in label_map.items()}

# -------------------------
# BUFFERS
# -------------------------
sequence = []
pred_buffer = deque(maxlen=SMOOTHING_WINDOW)

# -------------------------
# MEDIAPIPE
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "..."        # default — overwritten below if prediction is confident
    confidence = 0.0

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        features = build_feature_vector(result)
        sequence.append(features)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 127)

            probs = model.predict(input_data, verbose=0)[0]
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))

            print(f"PRED: {pred}  CONF: {confidence:.3f}")

            pred_buffer.append(pred)
            final_pred = max(set(pred_buffer), key=pred_buffer.count)

            if pred_buffer.count(final_pred) > MIN_VOTES and confidence > CONF_THRESHOLD:
                text = inv_label_map.get(final_pred, "Unknown")

    else:
        if sequence:
            sequence.pop(0)  # drain stale frames when hand disappears

    cv2.putText(
        frame,
        f"{text} ({confidence:.2f})",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()