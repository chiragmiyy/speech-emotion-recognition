# 🎤 Speech Emotion Recognition

A machine learning and deep learning-based system for recognizing emotions from speech using audio features like MFCCs, Spectrograms, and more.

## 📌 Overview

This project implements a Speech Emotion Recognition (SER) pipeline that uses audio signal processing and classification algorithms to detect emotions from speech. It supports multiple datasets, feature extractors, classifiers, and evaluation metrics.

---

## 🧠 Supported Emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Pleasant Surprise
- Boredom

---

## 🛠️ Features

- 🔉 Extracts audio features (MFCC, Chromagram, Spectrogram, etc.)
- 🤖 Classifiers: SVC, RandomForest, GradientBoosting, KNeighbors, MLP, RNN
- 🧪 Hyperparameter tuning via GridSearchCV
- 📊 Evaluation: Accuracy, Confusion Matrix
- 💾 Model saving & loading (`.pkl`)
- 🔍 Dataset support: RAVDESS, TESS, EMO-DB, Custom

---

## 📦 Tech Stack

| Domain | Tools |
|--------|-------|
| Programming | Python |
| Audio Processing | Librosa, OpenSMILE |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch, HuggingFace Transformers (Wav2Vec2) |
| Deployment (Optional) | Firebase Functions, Streamlit, Gradio |

---

## 📁 Project Structure

```bash
emotion-recognition-using-speech/
├── data/                   # Audio files and datasets
├── models/                 # Saved models (.pkl, .pt)
├── notebooks/              # Jupyter exploration files
├── src/                    # Core logic
│   ├── features/           # Feature extraction
│   ├── models/             # Training logic
│   ├── utils/              # Helper functions
├── results/                # Plots, confusion matrices
├── LICENSE
├── README.md
├── CITATION.cff
```

---

## 🚀 Getting Started

1. Clone the repo

git clone https://github.com/chirgamiyy/speech-emotion-recognition.git
cd speech-emotion-recognition

2. Install dependencies
   
pip install -r requirements.txt

3. Run training or prediction
   
python src/train.py        # Train model
python src/predict.py      # Predict emotion from audio

---

## 📊 Example Results

### 🔹 Model Accuracy Comparison (93.96%)
![Model Accuracy](./results/model_accuracy_comparison.png)

### 🔹 Confusion Matrix (on Combined Dataset)
![Confusion Matrix](./results/confusion_matrix.png)

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙌 Acknowledgements
  
- **Audio Datasets**:
  - [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
  - [TESS on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
  - [CREMA-D on Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad)

- **Feature Extraction Libraries**:
  - [Librosa](https://librosa.org/)
  - [OpenSMILE Toolkit](https://audeering.github.io/opensmile/)

- **Machine Learning & Deep Learning**:
  - [Scikit-learn](https://scikit-learn.org/)
  - [PyTorch](https://pytorch.org/)
  - [HuggingFace Transformers](https://huggingface.co/)

> If you build upon this work, please consider citing it via the [`CITATION.cff`](./CITATION.cff) file.

---

## 📚 Citation

If you use this work, please cite it using the metadata in [`CITATION.cff`](./CITATION.cff).

```bibtex
@software{agrawal_2025_ser,
  author = {Chirag Agrawal},
  title = {Speech Emotion Recognition},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/chirgamiyy/speech-emotion-recognition}
}
```
Feel free to ⭐ the repo if you found it helpful!
