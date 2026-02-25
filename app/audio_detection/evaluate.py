# ./app/audio_detection/evaluate.py

import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .config import FEATURES_FILE, BATCH_SIZE, MODEL_NAME
from .dataset import AudioDataset
from .models import CNN_LSTM, TCN, TCN_LSTM

def main():
    # Load dataset
    df = pd.read_csv(FEATURES_FILE)
    dataset = AudioDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if MODEL_NAME == 'cnn-lstm':
        model = CNN_LSTM(input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    elif MODEL_NAME == 'tcn':
        model = TCN(input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    elif MODEL_NAME == 'tcn-lstm':
        model = TCN_LSTM(input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    else:
        raise ValueError("MODEL_NAME must be 'cnn-lstm', 'tcn', or 'tcn-lstm'")
    
    model = model.to(device)
    model.load_state_dict(torch.load(f'./models/{MODEL_NAME}_audio_classifier.pth', map_location=device))
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # If y is a tuple of strings from dataset.py since we haven't encoded them
            if isinstance(y, tuple) or isinstance(y, list):
                y_tensor = torch.tensor([1 if label == 'FAKE' else 0 for label in y]).to(device)
            else:
                y_tensor = y.to(device)
                
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_tensor.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    main()