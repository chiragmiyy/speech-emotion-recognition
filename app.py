import os
from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import logging
from flask_cors import CORS

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load model
model = joblib.load("models/emotion_classifier.pkl")

# Flask app
app = Flask(__name__)
CORS(app)  # enable CORS if using with frontend

# Ensure temp folder exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        features = np.hstack([mfccs, chroma, zcr, rms])
        return features
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(TEMP_DIR, "temp_audio.wav")
    file.save(file_path)

    features = extract_features(file_path)
    os.remove(file_path)

    if features is None:
        return jsonify({'error': 'Could not extract features'}), 500

    prediction = model.predict([features])[0]
    return jsonify({'status': 'success', 'emotion': prediction})

if __name__ == '__main__':
    app.run(debug=True)