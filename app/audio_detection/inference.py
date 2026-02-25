# ./app/audio_detection/inference.py

import torch
from pydub import AudioSegment
import numpy as np
from scipy import signal
import soundfile as sf

from .config import SAMPLE_RATE, CLIP_LENGTH
from .models import CNN_LSTM, TCN, TCN_LSTM

MODEL_NAME = 'cnn-lstm'  # same options: 'cnn-lstm', 'tcn', 'tcn-lstm'

def predict(file_path, model, device):
    chunk_size = CLIP_LENGTH
    preds = []
    
    # Read the large file in small streams/blocks without loading fully to RAM
    for block in sf.blocks(file_path, blocksize=chunk_size, fill_value=0):
        # block is raw values, convert to target shape
        x = torch.tensor(block, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
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

# You may adjust input_dim if using a dataset input_dim
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