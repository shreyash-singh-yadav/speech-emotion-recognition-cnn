# speech-emotion-recognition-cnn
Speech Emotion Recognition using CNN on RAVDESS dataset
# Speech Emotion Recognition using CNN

This project implements a Speech Emotion Recognition (SER) system using a
Convolutional Neural Network (CNN) trained on the RAVDESS dataset.

## Dataset
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Actor-independent split
  - Training actors: 01–20
  - Testing actors: 21–24

## Emotions Classified
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Feature Extraction
- Log-Mel Spectrograms
- Sampling rate: 22,050 Hz
- Mel bands: 128
- Fixed length (padding/truncation)

## Model Architecture
- 2D CNN
- Convolution + MaxPooling + Dropout layers
- Global Average Pooling
- Softmax output (8 classes)

## Files
- `parse_and_split.py` – Dataset parsing and actor-wise split
- `extract_features.py` – Feature extraction (log-mel spectrograms)
- `train_cnn.py` – CNN training and evaluation
- `emotion_cnn.keras` – Trained model
- `requirements.txt` – Dependencies

## How to Run
```bash
pip install -r requirements.txt
python parse_and_split.py
python extract_features.py
python train_cnn.py
Result
Test accuracy: ~50%
