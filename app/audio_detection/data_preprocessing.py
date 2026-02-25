# ./app/deepfake_detection/data_preprocessing.py

import os
import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset
from pydub import AudioSegment  # Handles mp3

from .config import REAL_DIR, FAKE_DIR, SEGMENTS_DIR, CLIP_LENGTH, SAMPLE_RATE, WINDOW_SIZE, RANDOM_STATE

# Create segment folders
os.makedirs(os.path.join(SEGMENTS_DIR, "REAL"), exist_ok=True)
os.makedirs(os.path.join(SEGMENTS_DIR, "FAKE"), exist_ok=True)

# --- Load audio ---
def load_audio(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'mp3':
        audio = AudioSegment.from_mp3(file_path)
        audio = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
    else:
        import soundfile as sf
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    return audio, sr

def resample_to_16k(audio, sr):
    if sr == SAMPLE_RATE:
        return audio
    new_len = int(round(len(audio) * SAMPLE_RATE / sr))
    return signal.resample(audio, new_len)

# --- Segmenting ---
def split_and_save(input_dir, output_subdir):
    count = 0
    for fname in os.listdir(input_dir):
        if not fname.endswith(('.wav','.mp3')):
            continue
        path = os.path.join(input_dir, fname)
        audio, sr = load_audio(path)
        audio = resample_to_16k(audio, sr)
        num_segments = len(audio) // CLIP_LENGTH
        for i in range(num_segments):
            start = i * CLIP_LENGTH
            seg = audio[start:start+CLIP_LENGTH]
            seg_fname = f"{fname.replace('.wav','').replace('.mp3','')}_seg{i}.wav"
            seg_path = os.path.join(SEGMENTS_DIR, output_subdir, seg_fname)
            import soundfile as sf
            sf.write(seg_path, seg, SAMPLE_RATE)
            count += 1
    print(f"{count} segments saved from {input_dir}.")

def preprocess_audio():
    # Instead of extracting segments, just create a DataFrame of the raw files
    records = []
    
    # Process REAL files
    if os.path.exists(REAL_DIR):
        for fname in os.listdir(REAL_DIR):
            if fname.endswith(('.wav', '.mp3')):
                records.append({
                    'filepath': os.path.join(REAL_DIR, fname),
                    'label': 'REAL'
                })
                
    # Process FAKE files
    if os.path.exists(FAKE_DIR):
        for fname in os.listdir(FAKE_DIR):
            if fname.endswith(('.wav', '.mp3')):
                records.append({
                    'filepath': os.path.join(FAKE_DIR, fname),
                    'label': 'FAKE'
                })
                
    df = pd.DataFrame(records)
    
    # Save the dataframe for training
    os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)
    print(f"Created metadata CSV with {len(df)} files at {FEATURES_FILE}")
    
    return df