import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

from models import CNNLSTMSubtypeECG

TRAIN_FILE = "train_abnormal.npz"
MODEL_OUT = "cnn_lstm_subtype.pt"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

CLASS_NAMES = [
    "Hypertrophy",
    "Electrical",
    "Other"
]


class ECGDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / len(loader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ECGDataset(TRAIN_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Training dataset size:", len(dataset))
    print("Example X shape:", dataset[0][0].shape)
    print("Example y:", dataset[0][1].item())

    y_np = dataset.y.numpy()
    num_classes = 3

    unique, counts = np.unique(y_np, return_counts=True)
    print("Subtype training label counts:")
    for u, c in zip(unique, counts):
        print(f"{CLASS_NAMES[u]}: {c}")

    model = CNNLSTMSubtypeECG(
        in_channels=dataset.X.shape[1],
        num_classes=num_classes,
        lstm_hidden=128,
        lstm_layers=1
    ).to(device)

    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_np)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()