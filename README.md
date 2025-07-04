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

git clone https://github.com/chirgamiyy/spech-emotion-recognition.git
cd spech-emotion-recognition

2. Install dependencies
   
pip install -r requirements.txt

3. Run training or prediction
   
python src/train.py        # Train model
python src/predict.py      # Predict emotion from audio

---

## 📊 Example Results

| Model                 | Dataset              | Accuracy |
|----------------------|----------------------|----------|
| RandomForest          | RAVDESS              | 84.7%    |
| Wav2Vec2 (Fine-tuned) | TESS + EMO-DB        | 88.9%    |
| MLPClassifier         | Custom               | 81.2%    |
| 🔥 **Best Model**     | Combined Datasets    | **93.96%** ✅ |

- The best model achieves **93.96% accuracy** on a balanced, merged dataset of RAVDESS, TESS, EMO-DB, and custom speech recordings.
- All models are evaluated using standard metrics like accuracy, F1-score, and confusion matrix.
- Visualizations and training logs are available in the [`/results/`](./results/) folder.

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙌 Acknowledgements

- Based on original work by [@x4nth055](https://github.com/x4nth055/emotion-recognition-using-speech)
- **Audio Datasets**:
  - [RAVDESS](https://zenodo.org/record/1188976)
  - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
  - [EMO-DB (Berlin Emotional Database)](http://emodb.bilderbar.info/)
- **Feature Extraction Libraries**:
  - [Librosa](https://librosa.org/)
  - [OpenSMILE Toolkit](https://audeering.github.io/opensmile/)
- **Machine Learning & Deep Learning**:
  - [Scikit-learn](https://scikit-learn.org/)
  - [PyTorch](https://pytorch.org/)
  - [HuggingFace Transformers](https://huggingface.co/)

If you build upon this work, please consider citing it via the [`CITATION.cff`](./CITATION.cff) file.

---

## 📚 Citation

If you use this work, please cite it using the metadata in [`CITATION.cff`](./CITATION.cff).

```bibtex
@software{agrawal_2025_ser,
  author = {Chirag Agrawal},
  title = {Speech Emotion Recognition},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/chirgamiyy/spech-emotion-recognition}
}
```
Feel free to ⭐ the repo if you found it helpful!
