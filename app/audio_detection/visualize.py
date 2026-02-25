# ./app/audio_detection/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd
from torch.utils.data import DataLoader

from .config import FEATURES_FILE, BATCH_SIZE
from .dataset import AudioDataset
from .models import CNN_LSTM

# Load dataset
df = pd.read_csv(FEATURES_FILE)
dataset = AudioDataset(df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = CNN_LSTM(input_dim=dataset.input_dim, num_classes=dataset.num_classes).to(device)
model.load_state_dict(torch.load('./models/cnn-lstm_audio_classifier.pth', map_location=device))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        if isinstance(y, tuple) or isinstance(y, list):
            y_tensor = torch.tensor([1 if label == 'FAKE' else 0 for label in y]).to(device)
        else:
            y_tensor = y.to(device)
            
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_tensor.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
# plt.show() # save instead to avoid blocking script execution
plt.savefig('./confusion_matrix.png')
print("Saved confusion matrix to ./confusion_matrix.png")