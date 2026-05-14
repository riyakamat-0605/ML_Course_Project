from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import decode_image, extract_landmarks_from_image, predict_gesture
from tts import generate_speech
import os

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/')
def home():
    return jsonify({"message": "ISL Backend is running!"})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    try:
        img = decode_image(data['image'])
        landmarks = extract_landmarks_from_image(img)
        if landmarks is None:
            return jsonify({
                "label": None,
                "confidence": 0.0,
                "hand_detected": False
            })
        label, confidence = predict_gesture(landmarks)
        return jsonify({
            "label": label,
            "confidence": confidence,
            "hand_detected": True
        })
    except Exception as e:
        print(f"PREDICT ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    text = data.get('text', '')
    speed = data.get('speed', 'normal')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        filename = generate_speech(text, speed)
        return send_from_directory('static', filename, mimetype='audio/mpeg')
    except Exception as e:
        print(f"SPEAK ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')