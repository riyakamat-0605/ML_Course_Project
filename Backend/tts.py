from gtts import gTTS
import os
import uuid
from rnn_text_processor import process_text_with_rnn

def generate_speech(text, speed='normal'):
    # Step 1: Process text through RNN
    improved_text = process_text_with_rnn(text)
    print(f"RNN Input:  '{text}'")
    print(f"RNN Output: '{improved_text}'")

    # Step 2: Convert to speech using gTTS
    slow = True if speed == 'slow' else False
    filename = f"speech_{uuid.uuid4().hex}.mp3"
    os.makedirs('static', exist_ok=True)
    filepath = os.path.join('static', filename)

    tts = gTTS(text=improved_text, lang='en', slow=slow)
    tts.save(filepath)

    return filename

print("tts.py loaded successfully!")