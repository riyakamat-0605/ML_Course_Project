# Deployment Guide — ISL Sign Language Recognition

## How to Run This Project Locally

### Step 1 — Download the Project
Open terminal and run:
git clone https://github.com/ananya4186/isl-sign-language-recognition.git
cd isl-sign-language-recognition

### Step 2 — Install Python Libraries
pip install -r requirements.txt

### Step 3 — Add the AI Model Files
Place these files inside the backend/model/ folder:
- model.h5
- labels.json
- scaler.pkl
(These files are provided by Module 1 — Riya Kamat)

### Step 4 — Start the Backend Server
cd backend
python app.py

You should see: Running on http://127.0.0.1:5000

### Step 5 — Open the Frontend
Open VS Code
Right click on frontend/index.html
Click "Open with Live Server"

### Step 6 — Allow Webcam Access
When the browser asks for webcam permission, click "Allow"

### Step 7 — Use the App
- Show your hand to the webcam
- The recognised gesture will appear on screen
- Click "Speak" to hear the text read aloud
- Click "Clear" to reset

## Requirements
- Python 3.10 or above
- Google Chrome / Firefox / Edge
- Working webcam
- Internet connection
