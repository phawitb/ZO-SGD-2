import os
import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    OPTForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    TrainerCallback,
)
import evaluate
import config
from utils import load_and_split_dataset
import shutil
import csv
from datetime import datetime

# Set random seed
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
set_seed(config.SEED)

# Output directory setup
base_output_dir = "output"
os.makedirs(base_output_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_output_dir) if d.startswith("train_")]
next_run_id = max([int(d.split("_")[1]) for d in existing_runs], default=0) + 1
run_dir = os.path.join(base_output_dir, f"train_{next_run_id}")
os.makedirs(run_dir, exist_ok=True)
shutil.copy("config.py", os.path.join(run_dir, "config.py"))

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
model = OPTForSequenceClassification.from_pretrained(config.MODEL_PATH)

# Load and tokenize dataset
dataset = load_and_split_dataset(
    dataset_path=config.DATASET_PATH,
    total_size=config.TOTAL_DATASET,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    seed=config.SEED,
)

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Logging setup
log_csv_path = os.path.join(run_dir, "log_full_bp_train.csv")
with open(log_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "test_acc", "timestamp"])
    writer.writeheader()

# Custom callback for logging
class EpochLoggerCallback(TrainerCallback):
    def __init__(self):
        self.last_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.last_loss = logs["loss"]

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        train_result = trainer.evaluate(tokenized_dataset["train"])
        val_result = trainer.evaluate(tokenized_dataset["validation"])
        test_result = trainer.evaluate(tokenized_dataset["test"])

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "test_acc", "timestamp"])
            writer.writerow({
                "epoch": epoch,
                "loss": self.last_loss,
                "train_acc": train_result.get("eval_accuracy"),
                "val_acc": val_result.get("eval_accuracy"),
                "test_acc": test_result.get("eval_accuracy"),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

# Training configuration
training_args = TrainingArguments(
    output_dir=os.path.join(run_dir, "temp"),
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=config.LEARNING_RATE,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    num_train_epochs=config.NUM_EPOCHS,
    weight_decay=config.WEIGHT_DECAY,
    logging_dir=os.path.join(run_dir, "logs"),
    logging_steps=50,
    load_best_model_at_end=False,
    seed=config.SEED,
    data_seed=config.SEED,
    report_to="none",
    logging_first_step=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EpochLoggerCallback()],
)

# Train model
trainer.train()

print("Training complete.")
print(f"Log saved to: {log_csv_path}")
