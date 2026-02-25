# ./app/audio_detection/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from .config import FEATURES_FILE, BATCH_SIZE, LR, EPOCHS, RANDOM_STATE, MODEL_NAME
from .data_preprocessing import preprocess_audio
from .dataset import AudioDataset
from .models import CNN_LSTM, TCN, TCN_LSTM

def main():
    # 1️⃣ Preprocess audio locations
    preprocess_audio()  # saves raw metadata dataframe to features.csv
    
    # 2️⃣ Load dataset CSV
    df = pd.read_csv(FEATURES_FILE)  # columns: filepath,label
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['label']
    )
    
    train_dataset = AudioDataset(train_df)
    val_dataset = AudioDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3️⃣ Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if MODEL_NAME == 'cnn-lstm':
        model = CNN_LSTM(input_dim=train_dataset.input_dim, num_classes=len(le.classes_))
    elif MODEL_NAME == 'tcn':
        model = TCN(input_dim=train_dataset.input_dim, num_classes=len(le.classes_))
    elif MODEL_NAME == 'tcn-lstm':
        model = TCN_LSTM(input_dim=train_dataset.input_dim, num_classes=len(le.classes_))
    else:
        raise ValueError("MODEL_NAME must be 'cnn-lstm', 'tcn', or 'tcn-lstm'")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 4️⃣ Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
    
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
    
    # Save model
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), f'./models/{MODEL_NAME}_audio_classifier.pth')
    print(f"Training completed and {MODEL_NAME} model saved!")

if __name__ == '__main__':
    main()