import numpy as np

def build_feature_vector(result):

    primary_hand = [0] * 63
    uses_two_hands = 0

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]

        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

        primary_hand = data

        if len(result.multi_hand_landmarks) == 2:
            uses_two_hands = 1

    return np.array([uses_two_hands] + primary_hand + [0]*63)