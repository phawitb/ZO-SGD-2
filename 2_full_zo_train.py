import os
import random
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    OPTForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import config
import shutil
import csv
from datetime import datetime
from utils import load_and_split_dataset

# Set seed
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
set_seed(config.SEED)

# Output directory
base_output_dir = "output"
os.makedirs(base_output_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_output_dir) if d.startswith("train_")]
next_run_id = max([int(d.split("_")[1]) for d in existing_runs], default=0) + 1
run_dir = os.path.join(base_output_dir, f"train_{next_run_id}")
os.makedirs(run_dir, exist_ok=True)
shutil.copy("config.py", os.path.join(run_dir, "config.py"))

# Log setup
log_path = os.path.join(run_dir, "log_full_zo_train.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "test_acc", "timestamp"])
    writer.writeheader()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
model = OPTForSequenceClassification.from_pretrained(config.MODEL_PATH).to("cuda")

# Load dataset
dataset: DatasetDict = load_and_split_dataset(
    dataset_path=config.DATASET_PATH,
    seed=config.SEED,
    total_size=config.TOTAL_DATASET,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding='max_length')

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["sentence"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Dataloaders
train_loader = DataLoader(tokenized_dataset["train"], batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(tokenized_dataset["validation"], batch_size=config.BATCH_SIZE, collate_fn=data_collator)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=config.BATCH_SIZE, collate_fn=data_collator)

# ZO-SGD hyperparameters
mu = config.ZO_MU
lr = config.LEARNING_RATE
P = config.ZO_PERTURBATIONS

# Utilities
def params_to_vector(params):
    return torch.cat([p.view(-1) for p in params])

def vector_to_params(vector, params_template):
    idx = 0
    new_params = []
    for p in params_template:
        numel = p.numel()
        new_p = vector[idx:idx + numel].view_as(p).to(p.device)
        new_params.append(new_p)
        idx += numel
    return new_params

def update_model_params(model, new_params):
    with torch.no_grad():
        for p, new_p in zip(model.parameters(), new_params):
            p.copy_(new_p)

def compute_loss(model, batch):
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss.item()

metric = evaluate.load("accuracy")

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return metric.compute(predictions=all_preds, references=all_labels)

# ZO-SGD Training
print("\nTraining ZO-SGD")
params = list(model.parameters())

for epoch in range(config.NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}")
    epoch_losses = []

    for step, batch in enumerate(tqdm(train_loader)):
        model.eval()
        w = params_to_vector(params)
        grad_est = torch.zeros_like(w)
        losses = []

        for _ in range(P):
            u = torch.randn_like(w)
            w_pos = w + mu * u
            w_neg = w - mu * u

            theta_pos = vector_to_params(w_pos, params)
            theta_neg = vector_to_params(w_neg, params)

            update_model_params(model, theta_pos)
            L_pos = compute_loss(model, batch)

            update_model_params(model, theta_neg)
            L_neg = compute_loss(model, batch)

            grad_est += ((L_pos - L_neg) / (2 * mu)) * u
            losses.append((L_pos + L_neg) / 2)

        grad_est /= P
        avg_loss = np.mean(losses)
        grad_norm = grad_est.norm().item()
        epoch_losses.append(avg_loss)

        new_w = w - lr * grad_est
        new_params = vector_to_params(new_w, params)
        update_model_params(model, new_params)

        if (step + 1) % 50 == 0:
            print(f"Step {step+1} | Loss: {avg_loss:.4f} | Grad Norm: {grad_norm:.2f}")

    train_acc = evaluate_model(model, train_loader)["accuracy"]
    val_acc = evaluate_model(model, val_loader)["accuracy"]
    test_acc = evaluate_model(model, test_loader)["accuracy"]
    epoch_loss = np.mean(epoch_losses)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "test_acc", "timestamp"])
        writer.writerow({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })

# Save model and tokenizer
model.save_pretrained(os.path.join(run_dir, "opt-sst2-zo-finetuned"))
tokenizer.save_pretrained(os.path.join(run_dir, "opt-sst2-zo-finetuned"))
print(f"Model saved to {run_dir}/opt-sst2-zo-finetuned")
print(f"Logs saved to {log_path}")
