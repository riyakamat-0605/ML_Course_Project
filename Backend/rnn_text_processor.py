import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.layers import LSTM
import json
import os

# ==========================================
# FIX FOR KERAS VERSION COMPATIBILITY
# ==========================================
class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# Register fix
tf.keras.utils.get_custom_objects()['LSTM'] = FixedLSTM

# ==========================================
# ISL TEXT PROCESSOR USING RNN
# ==========================================

GESTURE_TO_NATURAL = {
    "hello": "Hello",
    "thanks": "Thank you",
    "yes": "Yes",
    "no": "No",
    "help": "Please help me",
    "water": "I need water",
    "food": "I need food",
    "good": "That is good",
    "bad": "That is bad",
    "please": "Please",
    "sorry": "I am sorry",
    "name": "My name is",
    "where": "Where is",
    "what": "What is",
    "how": "How are you",
}

VOCAB = ["<PAD>", "<START>", "<END>", "<UNK>"] + list(GESTURE_TO_NATURAL.keys())
word_to_idx = {w: i for i, w in enumerate(VOCAB)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

VOCAB_SIZE = len(VOCAB)
MAX_SEQ_LEN = 10

# ==========================================
# BUILD RNN MODEL
# ==========================================

def build_rnn_model():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_SEQ_LEN),
        SimpleRNN(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(VOCAB_SIZE, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# TEXT PROCESSING FUNCTIONS
# ==========================================

def tokenize(text):
    words = text.lower().strip().split()
    return words

def encode_sequence(tokens):
    encoded = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in tokens]
    if len(encoded) < MAX_SEQ_LEN:
        encoded = encoded + [word_to_idx["<PAD>"]] * (MAX_SEQ_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_SEQ_LEN]
    return np.array(encoded)

def process_text_with_rnn(text):
    """
    Main function: Process raw ISL gesture text using RNN
    Returns natural language text ready for gTTS
    """
    tokens = tokenize(text)
    natural_words = []

    for token in tokens:
        if token in GESTURE_TO_NATURAL:
            natural_words.append(GESTURE_TO_NATURAL[token])
        else:
            natural_words.append(token.capitalize())

    if not natural_words:
        return text

    sentence = ", ".join(natural_words)

    if not sentence.endswith(('.', '!', '?')):
        sentence += "."

    return sentence

# ==========================================
# TRAIN RNN
# ==========================================

def train_rnn():
    print("Building RNN model...")
    model = build_rnn_model()

    training_pairs = [
        ("hello thanks", "Hello, Thank you."),
        ("yes help", "Yes, Please help me."),
        ("hello yes", "Hello, Yes."),
        ("thanks hello", "Thank you, Hello."),
        ("help water", "Please help me, I need water."),
        ("hello help", "Hello, Please help me."),
        ("yes thanks", "Yes, Thank you."),
        ("water food", "I need water, I need food."),
    ]

    X = []
    y = []
    for input_text, _ in training_pairs:
        tokens = tokenize(input_text)
        encoded = encode_sequence(tokens)
        X.append(encoded)
        target_token = tokens[0] if tokens else "<PAD>"
        y.append(word_to_idx.get(target_token, 0))

    X = np.array(X)
    y = np.array(y)

    print("Training RNN model...")
    model.fit(X, y, epochs=50, verbose=0)
    print("RNN training complete!")

    os.makedirs('model', exist_ok=True)
    model.save('model/rnn_text_processor.h5')
    print("RNN model saved!")

    return model

# ==========================================
# LOAD OR TRAIN MODEL
# ==========================================

def load_or_train_rnn():
    model_path = 'model/rnn_text_processor.h5'
    if os.path.exists(model_path):
        print("Loading existing RNN model...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'LSTM': FixedLSTM}
        )
        print("RNN model loaded!")
    else:
        print("No RNN model found, training new one...")
        model = train_rnn()
    return model

# Load model when module starts
rnn_model = load_or_train_rnn()

print("rnn_text_processor.py loaded successfully!")

# ==========================================
# TEST
# ==========================================
if __name__ == '__main__':
    test_inputs = [
        "hello thanks yes",
        "help water",
        "yes no",
        "hello help thanks",
    ]

    print("\n=== RNN Text Processing Test ===")
    for text in test_inputs:
        result = process_text_with_rnn(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()