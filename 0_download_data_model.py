from datasets import load_dataset
from transformers import AutoTokenizer, OPTForSequenceClassification
import os

# === Save Root Folder ===
save_root = "source"  
dataset_path = os.path.join(save_root, "dataset")
model_path = os.path.join(save_root, "model")

# === Create Directories ===
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# === Load & Save SST-2 Dataset ===
dataset = load_dataset("glue", "sst2")
dataset.save_to_disk(dataset_path)
print(f"SST-2 dataset saved to {dataset_path}")

# === Load & Save OPT-125M Model + Tokenizer ===
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model and tokenizer saved to {model_path}")
