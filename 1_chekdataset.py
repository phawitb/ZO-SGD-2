from datasets import load_from_disk
from collections import Counter

# Load dataset
dataset_path = "/home/phawit/Documents/MyDisk/Z-New/source/dataset"
dataset = load_from_disk(dataset_path)

# Show label stats for each split
for split in dataset.keys():
    labels = dataset[split]["label"]
    print(f"Split: {split}")
    print(f"Label values: {set(labels)}")
    print(f"Count: {Counter(labels)}")
    print(f"Min: {min(labels)}, Max: {max(labels)}")
    print("-" * 50)
