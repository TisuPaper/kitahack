# ./app/audio_detection/__init__.py

"""
Audio Detection Package
----------------------
Contains modules for:
- config: configuration variables
- models: CNN-LSTM, TCN, and TCN-LSTM models
- inference (via app/audio_inference.py): multi-model ensemble prediction
"""

from .models import CNN_LSTM, TCN, TCN_LSTM

__all__ = [
    "CNN_LSTM",
    "TCN",
    "TCN_LSTM",
]