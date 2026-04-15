import h5py
import numpy as np
from sklearn.model_selection import train_test_split

H5_FILE = "ecg_dataset.h5"
TRAIN_OUT = "train_abnormal.npz"
TEST_OUT = "test_abnormal.npz"
TEST_SIZE = 0.2
RANDOM_STATE = 42

CLASS_NAMES = [
    "Hypertrophy",
    "Electrical",
    "Other"
]


def encode_abnormal_labels(y_raw):
    y_raw = np.array(y_raw)

    y_str = []
    for v in y_raw:
        if isinstance(v, bytes):
            y_str.append(v.decode("utf-8"))
        else:
            y_str.append(str(v))
    y_str = np.array(y_str)

    # Keep only abnormal samples
    abnormal_mask = y_str != "Normal"
    y_abn = y_str[abnormal_mask]

    # Stage 2 subtype labels:
    # 0 = Hypertrophy
    # 1 = Electrical (Atrial + Conduction)
    # 2 = Other (Other + Ventricular + Paced + Pre-excitation)
    label_map = {
        "Hypertrophy": 0,
        "Atrial": 1,
        "Conduction": 1,
        "Other": 2,
        "Ventricular": 2,
        "Paced": 2,
        "Pre-excitation": 2
    }

    y_encoded = np.array([label_map[label] for label in y_abn], dtype=np.int64)

    return abnormal_mask, y_encoded


def main():
    with h5py.File(H5_FILE, "r") as f:
        X = f["X"][:]
        y_raw = f["y"][:]

    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y_raw.shape)

    X = np.array(X, dtype=np.float32)
    abnormal_mask, y = encode_abnormal_labels(y_raw)
    X = X[abnormal_mask]

    print("Filtered abnormal X shape:", X.shape)
    print("Encoded subtype y shape:", y.shape)

    unique, counts = np.unique(y, return_counts=True)
    print("Subtype label counts:")
    for u, c in zip(unique, counts):
        print(f"{CLASS_NAMES[u]}: {c}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    np.savez(TRAIN_OUT, X=X_train, y=y_train)
    np.savez(TEST_OUT, X=X_test, y=y_test)

    print(f"Saved {TRAIN_OUT}")
    print(f"Saved {TEST_OUT}")
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()