import pandas as pd

file_path = "data/Indian Sign Language Gesture Landmarks.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nTarget Classes:")
print(df["target"].value_counts().sort_index())