# ./app/audio_detection/inference.py

import torch
from pydub import AudioSegment
import numpy as np
import torchaudio
from .config import SAMPLE_RATE, CLIP_LENGTH, MODEL_NAME
from .models import CNN_LSTM, TCN, TCN_LSTM

def predict(file_path, model, device):
    chunk_size = CLIP_LENGTH
    preds = []
    
    # Check if file exists to prevent hard crash
    import os
    if not os.path.exists(file_path):
        print(f"Warning: Demo file {file_path} not found.")
        return 0

    info = torchaudio.info(file_path)
    total_frames = info.num_frames
    original_sr = info.sample_rate
    
    # Calculate native chunk size before resampling
    native_chunk_size = int(chunk_size * (original_sr / SAMPLE_RATE))
    
    for start_frame in range(0, total_frames, native_chunk_size):
        # Load exactly that chunk from disk
        waveform, sr = torchaudio.load(
            file_path, 
            frame_offset=start_frame, 
            num_frames=native_chunk_size
        )
        
        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad if short
        if waveform.shape[1] < chunk_size:
            waveform = torch.nn.functional.pad(waveform, (0, chunk_size - waveform.shape[1]))
        elif waveform.shape[1] > chunk_size:
            waveform = waveform[:, :chunk_size]
            
        x = waveform.unsqueeze(0).to(device) # [1, 1, 32000]
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            pred = outputs.argmax(dim=1).item()
            preds.append(pred)
            
    if not preds:
        return 0

    # Aggregate predictions over the long file length
    fake_count = sum(preds)
    is_fake = (fake_count / len(preds)) > 0.5 # If > 50% of the stream is fake
    return 1 if is_fake else 0

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 1  # for raw audio single channel
if MODEL_NAME == 'cnn-lstm':
    model = CNN_LSTM(input_dim=input_dim, num_classes=2)
elif MODEL_NAME == 'tcn':
    model = TCN(input_dim=input_dim, num_classes=2)
elif MODEL_NAME == 'tcn-lstm':
    model = TCN_LSTM(input_dim=input_dim, num_classes=2)
else:
    raise ValueError("MODEL_NAME must be 'cnn-lstm', 'tcn', or 'tcn-lstm'")

model = model.to(device)
model.load_state_dict(torch.load(f'./models/{MODEL_NAME}_audio_classifier.pth', map_location=device))

# Example usage
file_path = './app/data/DEMONSTRATION/linus-to-musk-DEMO.mp3'
pred = predict(file_path, model, device)
label_map = {0:'REAL', 1:'FAKE'}
print(f"Prediction for {file_path}: {label_map[pred]}")