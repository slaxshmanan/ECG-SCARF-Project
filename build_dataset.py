import os
import wfdb
import neurokit2 as nk
import numpy as np
from sklearn.model_selection import train_test_split
import torch

DATA_DIR = "data/mitdb/mit-bih-arrhythmia-database-1.0.0"

window_size = 300
half_window = window_size // 2

all_beats = []
all_labels = []

record_ids = sorted({
    f.split(".")[0]
    for f in os.listdir(DATA_DIR)
    if f.endswith(".dat")
})

print("Found records:", record_ids)

for record_id in record_ids:
    record_path = os.path.join(DATA_DIR, record_id)
    print(f"Processing record {record_id}...")

    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")

        signal = record.p_signal[:, 0]
        fs = record.fs

        cleaned = nk.ecg_clean(signal, sampling_rate=fs)

        ann_samples = np.array(annotation.sample)
        ann_symbols = np.array(annotation.symbol)

        for i in range(len(ann_samples)):
            center = ann_samples[i]
            label = ann_symbols[i]

            start = center - half_window
            end = center + half_window

            if start < 0 or end > len(cleaned):
                continue

            beat = cleaned[start:end]
            if len(beat) != window_size:
                continue

            # Binary classification:
            # 0 = Normal
            # 1 = Abnormal
            if label == "N":
                binary_label = 0
            else:
                binary_label = 1

            all_beats.append(beat)
            all_labels.append(binary_label)

    except Exception as e:
        print(f"Skipping {record_id} due to error: {e}")

all_beats = np.array(all_beats, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int64)

print("\nDataset summary:")
print("Beats shape:", all_beats.shape)
print("Class distribution [Normal, Abnormal]:", np.bincount(all_labels))

X_train, X_test, y_train, y_test = train_test_split(
    all_beats,
    all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("\nFinal tensor shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

torch.save(
    {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    },
    "dataset_binary.pt"
)

print("\nSaved dataset to dataset_binary.pt")