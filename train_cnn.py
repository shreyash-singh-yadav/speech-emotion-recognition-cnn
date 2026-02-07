import numpy as np

# -----------------------
# Load extracted features
# -----------------------
from extract_features import X_train, y_train, X_test, y_test

# -----------------------
# Imports
# -----------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Encode labels
# -----------------------
le = LabelEncoder()

y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

y_train_cat = to_categorical(y_train_enc, num_classes=8)
y_test_cat  = to_categorical(y_test_enc, num_classes=8)

# -----------------------
# CNN model
# -----------------------
model = Sequential([
    Input(shape=(128, 128, 1)),

    Conv2D(32, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.4),

    GlobalAveragePooling2D(),

    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(8, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------
# Train
# -----------------------
history = model.fit(
    X_train,
    y_train_cat,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[
        EarlyStopping(
            patience=6,
            restore_best_weights=True,
            monitor="val_accuracy"
        )
    ]
)

# -----------------------
# Evaluate
# -----------------------
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print("Test accuracy:", test_acc)

# -----------------------
# Save model
# -----------------------
model.save("emotion_cnn.keras")
print("Model saved as emotion_cnn.keras")
