# ./app/audio_detection/config.py

import os

# Local dataset paths
DATA_DIR = "./app/data/DeepVoice/KAGGLE/AUDIO"
REAL_DIR = os.path.join(DATA_DIR, "REAL")
FAKE_DIR = os.path.join(DATA_DIR, "FAKE")
CSV_FILE = os.path.join(DATA_DIR, "DATASET-balanced.csv")
SEGMENTS_DIR = os.path.join(DATA_DIR, "SEGMENTS")
FEATURES_FILE = os.path.join(SEGMENTS_DIR, "features.csv")

# Audio settings
SAMPLE_RATE = 16000
CLIP_DURATION = 2   # seconds
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION

# Preprocessing
WINDOW_SIZE = 5
RANDOM_STATE = 42

# Training
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30