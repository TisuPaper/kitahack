import torch
import torchaudio
import os
import pandas as pd
from torch.utils.data import Dataset
from .config import CLIP_LENGTH, SAMPLE_RATE

class AudioDataset(Dataset):
    def __init__(self, dataframe, clip_length=CLIP_LENGTH):
        self.df = dataframe
        self.clip_length = clip_length
        self.input_dim = 1 # 1 channel for raw audio
        
        # Determine number of classes from dataframe labels
        if 'label' in self.df.columns:
            self.num_classes = len(self.df['label'].unique())
        else:
            self.num_classes = 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.iloc[idx]['filepath']
        label = self.df.iloc[idx]['label']
        
        # Fallback if file does not exist
        if not os.path.exists(file_path):
            return torch.zeros((1, self.clip_length)), label

        try:
            # Get metadata to find the total frames (without reading the audio)
            info = torchaudio.info(file_path)
            total_frames = info.num_frames
            sr = info.sample_rate
            
            # Since torchaudio gets the frame count in its native sample rate,
            # we need to adjust the clip length to match the native SR before loading,
            # then resample the loaded chunk to our target SAMPLE_RATE (16000)
            
            native_clip_length = int(self.clip_length * (sr / SAMPLE_RATE))
            
            # Pick a random window
            if total_frames > native_clip_length:
                start_frame = torch.randint(0, total_frames - native_clip_length, (1,)).item()
            else:
                start_frame = 0
                
            # Load EXACTLY that chunk from disk
            waveform, original_sr = torchaudio.load(
                file_path, 
                frame_offset=start_frame, 
                num_frames=native_clip_length
            )
            
            # Resample to common 16000 Hz if necessary
            if original_sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)

            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad if the audio chunk is shorter than target length
            if waveform.shape[1] < self.clip_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.clip_length - waveform.shape[1]))
            elif waveform.shape[1] > self.clip_length:
                # Truncate if slightly longer due to rounding
                waveform = waveform[:, :self.clip_length]

            return waveform, label
            
        except Exception as e:
            # If any failure loading audio, return silence
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((1, self.clip_length)), label
