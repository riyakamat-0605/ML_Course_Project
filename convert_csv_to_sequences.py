import pandas as pd
import numpy as np
import os

CSV_PATH = "data/Indian Sign Language Gesture Landmarks.csv"
OUTPUT_PATH = "data/sequences_csv"
SEQUENCE_LENGTH = 30

os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Separate features and label
labels = df["target"].values
features = df.drop(columns=["target"]).values

print("Total rows:", len(df))

# Group by class
unique_labels = np.unique(labels)

for label in unique_labels:
    print(f"\nProcessing class {label}")

    # get all rows for this class
    class_data = features[labels == label]

    # create folder
    class_path = os.path.join(OUTPUT_PATH, str(label))
    os.makedirs(class_path, exist_ok=True)

    count = 0

    # split into sequences
    for i in range(0, len(class_data) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
        sequence = class_data[i:i+SEQUENCE_LENGTH]

        if sequence.shape == (SEQUENCE_LENGTH, 127):
            np.save(os.path.join(class_path, f"{count}.npy"), sequence)
            count += 1

    print(f"Saved {count} sequences for class {label}")

print("\n✅ CSV converted to sequences!")