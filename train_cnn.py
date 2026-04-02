import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)
import numpy as np

# -----------------------------
# 1. Load dataset
# -----------------------------
data = torch.load("dataset_binary.pt")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Loaded dataset:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# -----------------------------
# 2. Create DataLoaders
# -----------------------------
batch_size = 256

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 3. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 4. Class weights
# -----------------------------
class_counts = torch.bincount(y_train)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

print("Class counts [Normal, Abnormal]:", class_counts)
print("Class weights:", class_weights)

# -----------------------------
# 5. Class names
# -----------------------------
class_names = {
    0: "Normal",
    1: "Abnormal"
}

# -----------------------------
# 6. Define CNN model
# -----------------------------
class ECGCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Input length = 300
        # 300 -> 150 -> 75 -> 37
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 37, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ECGCNN(num_classes=2).to(device)
print(model)

# -----------------------------
# 7. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 8. Training loop
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

# -----------------------------
# 9. Save model
# -----------------------------
torch.save(model.state_dict(), "cnn_model_binary.pt")
print("\nSaved model to cnn_model_binary.pt")

# -----------------------------
# 10. Evaluation
# -----------------------------
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"], zero_division=0))

# -----------------------------
# 10.1 F1 score per class
# -----------------------------
f1_per_class = f1_score(all_labels, all_preds, average=None)

print("\nF1 Score Per Class:")
for i, f1 in enumerate(f1_per_class):
    print(f"{class_names[i]}: {f1:.4f}")

# -----------------------------
# 10.2 Confusion matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# 10.3 Sensitivity (Recall)
# -----------------------------
print("\nSensitivity (Recall) Per Class:")
for i in range(len(cm)):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{class_names[i]}: {sensitivity:.4f}")

# -----------------------------
# 10.4 Specificity
# -----------------------------
print("\nSpecificity Per Class:")
for i in range(len(cm)):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fn + fp)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"{class_names[i]}: {specificity:.4f}")

# -----------------------------
# 10.5 AUC-ROC
# -----------------------------
# Use probability of abnormal class (class 1)
abnormal_probs = all_probs[:, 1]

try:
    auc_score = roc_auc_score(all_labels, abnormal_probs)
    print(f"\nAUC-ROC: {auc_score:.4f}")
except Exception as e:
    print("\nCould not compute AUC-ROC:", e)