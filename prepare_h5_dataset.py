import h5py
import numpy as np
from sklearn.model_selection import train_test_split

H5_FILE = "ecg_dataset.h5"   # change this
TRAIN_OUT = "train_real.npz"
TEST_OUT = "test_real.npz"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def encode_labels(y_raw):
    y_raw = np.array(y_raw)

    # Decode bytes if needed
    y_str = []
    for v in y_raw:
        if isinstance(v, bytes):
            y_str.append(v.decode("utf-8"))
        else:
            y_str.append(str(v))
    y_str = np.array(y_str)

    classes = sorted(np.unique(y_str))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)

    print("Classes found:")
    for c, i in class_to_idx.items():
        print(f"  {i}: {c}")

    return y, class_to_idx


def main():
    with h5py.File(H5_FILE, "r") as f:
        X = f["X"][:]
        y_raw = f["y"][:]

    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y_raw.shape)

    X = np.array(X, dtype=np.float32)
    y, class_to_idx = encode_labels(y_raw)

    print("Encoded y shape:", y.shape)
    print("Label counts:", dict(zip(*np.unique(y, return_counts=True))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
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