# ./app/audio_detection/models.py

import torch
import torch.nn as nn

# -------------------------
# 1️⃣ CNN-LSTM
# -------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch, 1, seq_len] (from dataset)
        # No need to permute if it already is [batch, channels, seq_len]
        # But if it is [batch, seq_len, channels], we permute
        if x.shape[1] > x.shape[2]:
            x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        x, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logits for CrossEntropyLoss
        return x


# -------------------------
# 2️⃣ TCN
# -------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 1, dilation_size, padding, dropout))
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_channels[-1], 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logits for CrossEntropyLoss
        return x


# -------------------------
# 3️⃣ TCN-LSTM hybrid
# -------------------------
class TCN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes=2, tcn_channels=[64, 128, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        # TCN backbone
        layers = []
        for i in range(len(tcn_channels)):
            in_ch = input_dim if i == 0 else tcn_channels[i - 1]
            out_ch = tcn_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 1, dilation_size, padding, dropout))
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # LSTM on top
        self.lstm = nn.LSTM(tcn_channels[-1], 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.tcn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        x, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x