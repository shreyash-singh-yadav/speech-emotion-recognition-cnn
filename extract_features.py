import os
import numpy as np
import librosa

# -----------------------
# Parameters
# -----------------------
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_LEN = 128  # time frames

# -----------------------
# Feature extraction
# -----------------------
def extract_log_mel(file_path,augment=False):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if augment:
        if np.random.rand() < 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
        if np.random.rand() < 0.5:
            y = librosa.effects.pitch_shift(y, sr, n_steps=np.random.randint(-2, 3))
        if np.random.rand() < 0.5:
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    if log_mel.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :MAX_LEN]

    return log_mel

    # Fix length (important for CNN)
    if log_mel.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :MAX_LEN]

    return log_mel


# -----------------------
# Load dataset from previous step
# -----------------------
from parse_and_split import train_samples, test_samples

# -----------------------
# Build feature arrays
# -----------------------
X_train = np.array([extract_log_mel(fp) for fp, _ in train_samples])
y_train = np.array([label for _, label in train_samples])

X_test = np.array([extract_log_mel(fp) for fp, _ in test_samples])
y_test = np.array([label for _, label in test_samples])

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

