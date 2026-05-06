import joblib

model = joblib.load("models/isl_model.pkl")

print("Model loaded successfully!")
print(model)