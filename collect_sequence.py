import cv2
import mediapipe as mp
import numpy as np
import os

# 🔥 Dynamic label input (NO hardcoding)
LABEL = input("Enter gesture name (e.g., hello, thanks, yes): ").strip()

SAVE_PATH = f"data/sequences/{LABEL}"
SEQUENCE_LENGTH = 30

# Create folder if not exists
os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence = []
count = len(os.listdir(SAVE_PATH))  # continue numbering safely

def extract_features(result):
    left = [0] * 63
    right = [0] * 63
    uses_two_hands = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = result.multi_handedness[idx].classification[0].label

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            if label == "Left":
                left = data
            else:
                right = data

        if len(result.multi_hand_landmarks) == 2:
            uses_two_hands = 1

    return [uses_two_hands] + left + right  # 127 features


print(f"\n📸 Collecting sequences for: {LABEL}")
print("Press 'q' to stop\n")
print(f"Saving to folder: {SAVE_PATH}\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = extract_features(result)
        sequence.append(features)

        # Save after 30 frames
        if len(sequence) == SEQUENCE_LENGTH:
            np.save(os.path.join(SAVE_PATH, f"{count}.npy"), sequence)
            print(f"✅ Saved sequence {count}")

            sequence = []
            count += 1

    # Display info
    cv2.putText(frame, f"{LABEL} | Saved: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Sequence Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()