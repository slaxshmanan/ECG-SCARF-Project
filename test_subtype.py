import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)

from models import CNNLSTMSubtypeECG

TEST_FILE = "test_abnormal.npz"
MODEL_FILE = "cnn_lstm_subtype.pt"
BATCH_SIZE = 32

NUM_CLASSES = 3
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ECGDataset(TEST_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Test dataset size:", len(dataset))
    print("Example X shape:", dataset[0][0].shape)
    print("Unique labels present in this test split:", torch.unique(dataset.y).tolist())

    model = CNNLSTMSubtypeECG(
        in_channels=dataset.X.shape[1],
        num_classes=NUM_CLASSES,
        lstm_hidden=128,
        lstm_layers=1
    ).to(device)

    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_true, all_preds)
    f1_per_class = f1_score(all_true, all_preds, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
    f1_macro = f1_score(all_true, all_preds, average="macro", labels=np.arange(NUM_CLASSES), zero_division=0)
    f1_weighted = f1_score(all_true, all_preds, average="weighted", labels=np.arange(NUM_CLASSES), zero_division=0)

    cm = confusion_matrix(all_true, all_preds, labels=np.arange(NUM_CLASSES))

    try:
        auc_roc = roc_auc_score(all_true, all_probs, multi_class="ovr", labels=np.arange(NUM_CLASSES))
    except ValueError:
        auc_roc = None

    print("\n--- Subtype Test Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")

    print("\nF1 score per class:")
    for i, score in enumerate(f1_per_class):
        print(f"{CLASS_NAMES[i]}: {score:.4f}")

    if auc_roc is not None:
        print(f"\nAUC-ROC (OvR): {auc_roc:.4f}")
    else:
        print("\nAUC-ROC: Could not compute")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            all_true,
            all_preds,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()