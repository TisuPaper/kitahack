# ./app/audio_detection/__init__.py

"""
Audio Detection Package
----------------------
Contains modules for:
- config: configuration variables
- data_preprocessing: audio segmenting and preprocessing
- dataset: PyTorch Dataset for audio clips
- models: CNN-LSTM, TCN, and TCN-LSTM models
- train: training scripts
- evaluate: model evaluation
- inference: predicting new audio clips
- visualize: visualization of results
"""

# Optional: Expose key classes/functions at package level
from .config import *
from .data_preprocessing import *
from .dataset import *
from .models import *

__all__ = [
    "config",
    "data_preprocessing",
    "dataset",
    "models",
    "train",
    "evaluate",
    "inference",
    "visualize"
]