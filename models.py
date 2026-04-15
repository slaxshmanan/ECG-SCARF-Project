import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMRealECG(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, lstm_hidden=128, lstm_layers=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (B, 2, 5000)
        x = self.pool1(F.relu(self.conv1(x)))   # (B, 32, 2500)
        x = self.pool2(F.relu(self.conv2(x)))   # (B, 64, 1250)
        x = self.pool3(F.relu(self.conv3(x)))   # (B, 128, 625)

        x = x.permute(0, 2, 1)                  # (B, 625, 128)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]                  # (B, hidden)
        x = self.fc(x)
        return x


class CNNLSTMSubtypeECG(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, lstm_hidden=128, lstm_layers=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (B, 2, 5000)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x