import os

dataset_path = r"C:\Users\acer\home\shreyash\AIML_PROJECTS\ravdess"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

samples = []

for actor in sorted(os.listdir(dataset_path)):
    actor_path = os.path.join(dataset_path, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            parts = file.split("-")
            emotion = emotion_map[parts[2]]
            actor_id = parts[6].split(".")[0]
            file_path = os.path.join(actor_path, file)
            samples.append((file_path, emotion, actor_id))

print("Total samples:", len(samples))

train_actors = {f"{i:02d}" for i in range(1, 21)}
test_actors  = {f"{i:02d}" for i in range(21, 25)}

train_samples = []
test_samples = []

for file_path, emotion, actor_id in samples:
    if actor_id in train_actors:
        train_samples.append((file_path, emotion))
    elif actor_id in test_actors:
        test_samples.append((file_path, emotion))

print("Training samples:", len(train_samples))
print("Testing samples:", len(test_samples))

# Sanity checks
assert len(train_samples) > 0
assert len(test_samples) > 0
assert len(train_samples) + len(test_samples) == len(samples)

print("parse_and_split.py SUCCESS")