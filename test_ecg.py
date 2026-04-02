import wfdb
import neurokit2 as nk
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# Load data
record_path = "data/mitdb/mit-bih-arrhythmia-database-1.0.0/100"
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, "atr")

signal = record.p_signal[:, 0]
fs = record.fs

# Clean ECG
cleaned = nk.ecg_clean(signal, sampling_rate=fs)

# Annotation data
ann_samples = np.array(annotation.sample)
ann_symbols = np.array(annotation.symbol)

# Window extraction
window_size = 300
half_window = window_size // 2

beats = []
labels = []

for i in range(len(ann_samples)):
    center = ann_samples[i]
    start = center - half_window
    end = center + half_window

    if start < 0 or end > len(cleaned):
        continue

    beat = cleaned[start:end]
    label = ann_symbols[i]

    beats.append(beat)
    labels.append(label)

beats = np.array(beats)
labels = np.array(labels)

print("Beats shape:", beats.shape)
print("Labels shape:", labels.shape)
print("First 20 labels:", labels[:20])

# Map labels to broader classes
label_map = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4
}

filtered_beats = []
filtered_labels = []

for i in range(len(labels)):
    if labels[i] in label_map:
        filtered_beats.append(beats[i])
        filtered_labels.append(label_map[labels[i]])

filtered_beats = np.array(filtered_beats)
filtered_labels = np.array(filtered_labels)

print("Before rare-class removal:")
print("Filtered shape:", filtered_beats.shape)
print("Class distribution:", np.bincount(filtered_labels))

# Remove classes with fewer than 2 samples
unique_classes, counts = np.unique(filtered_labels, return_counts=True)
valid_classes = unique_classes[counts >= 2]

mask = np.isin(filtered_labels, valid_classes)
filtered_beats = filtered_beats[mask]
filtered_labels = filtered_labels[mask]

print("After rare-class removal:")
print("Filtered shape:", filtered_beats.shape)
print("Class distribution:", np.bincount(filtered_labels))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    filtered_beats,
    filtered_labels,
    test_size=0.2,
    random_state=42,
    stratify=filtered_labels
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)