import os
from datasets import load_from_disk, DatasetDict
import pickle
import socket

def send_data_to(data, node: tuple[str, int]):
    """
    Send a Python object over TCP socket.
    """
    serialized = pickle.dumps(data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(node)
        s.sendall(serialized)

def load_and_split_dataset(dataset_path, seed, total_size=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Load and split a HuggingFace dataset into train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    full_dataset = dataset["train"].shuffle(seed=seed)

    if total_size and total_size < len(full_dataset):
        full_dataset = full_dataset.select(range(total_size))

    total = len(full_dataset)
    n_train = int(train_ratio * total)
    n_val = int(val_ratio * total)
    n_test = total - n_train - n_val

    print(f"Using {total} samples (train={n_train}, val={n_val}, test={n_test})")

    return DatasetDict({
        "train": full_dataset.select(range(n_train)),
        "validation": full_dataset.select(range(n_train, n_train + n_val)),
        "test": full_dataset.select(range(n_train + n_val, total)),
    })
