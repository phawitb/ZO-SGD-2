# === IMPORT และ INITIAL SETUP (เหมือนเดิม) ===
import os, random, csv, socket, struct, shutil
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
from datetime import datetime
from tqdm import tqdm
from datasets import DatasetDict
from transformers import (
    AutoTokenizer, OPTForSequenceClassification,
    DataCollatorWithPadding, set_seed
)
from torch.utils.data import DataLoader
import config
from utils import load_and_split_dataset

# === Seed ===
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
set_seed(config.SEED)

# === Output Dir ===
base_output_dir = "output"
os.makedirs(base_output_dir, exist_ok=True)
run_id = max([int(d.split("_")[1]) for d in os.listdir(base_output_dir) if d.startswith("train_")], default=0) + 1
run_dir = os.path.join(base_output_dir, f"train_{run_id}")
os.makedirs(run_dir, exist_ok=True)
shutil.copy("config.py", os.path.join(run_dir, "config.py"))
log_path = os.path.join(run_dir, "log_split.csv")

with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "test_acc", "timestamp"])
    writer.writeheader()

# === Load Model ===
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
model = OPTForSequenceClassification.from_pretrained(config.MODEL_PATH).to("cuda")
split_idx = config.SPLIT_LAYER_IDX
embed = model.model.decoder.embed_tokens
early_layers = model.model.decoder.layers[:split_idx]
params = list(embed.parameters()) + [p for l in early_layers for p in l.parameters()]
metric = evaluate.load("accuracy")

# === Load Dataset ===
dataset: DatasetDict = load_and_split_dataset(
    dataset_path=config.DATASET_PATH,
    seed=config.SEED,
    total_size=config.TOTAL_DATASET,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
)
def tokenize(batch): return tokenizer(batch["sentence"], truncation=True, padding='max_length', max_length=128)
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["sentence"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(tokenized_dataset["train"], batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader   = DataLoader(tokenized_dataset["validation"], batch_size=config.BATCH_SIZE, collate_fn=data_collator)
test_loader  = DataLoader(tokenized_dataset["test"], batch_size=config.BATCH_SIZE, collate_fn=data_collator)

# === Socket ===
HOST = config.NODE2_IP
PORT = config.NODE2_PORT
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

def safe_sendall(data):
    global sock
    try: sock.sendall(data)
    except:
        sock.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        sock.sendall(data)

def send_tensor(tensor):
    data = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
    safe_sendall(struct.pack("!I", len(data)) + data)

def send_label_tensor(labels):
    data = labels.detach().cpu().numpy().astype(np.int64).tobytes()
    safe_sendall(struct.pack("!I", len(data)) + data)

def recv_float():
    data = b''
    while len(data) < 8:
        packet = sock.recv(8 - len(data))
        if not packet: raise ConnectionError("Loss recv failed")
        data += packet
    return struct.unpack("!d", data)[0]

def recv_pred_batch(size):
    data = b''
    while len(data) < 4 * size:
        packet = sock.recv(4 * size - len(data))
        if not packet: raise ConnectionError("Pred recv failed")
        data += packet
    return list(struct.unpack(f'!{size}I', data))

# === Helper Functions ===
def forward_node1(input_ids):
    x = embed(input_ids)
    for layer in early_layers:
        x = layer(x)[0]
    return x

def params_to_vector(params): return torch.cat([p.view(-1) for p in params])
def vector_to_params(vector, template):
    idx, new_params = 0, []
    for p in template:
        numel = p.numel()
        new_params.append(vector[idx:idx+numel].view_as(p).to(p.device))
        idx += numel
    return new_params

def update_model_params(params, new_params):
    with torch.no_grad():
        for p, new_p in zip(params, new_params): p.copy_(new_p)

def check_param_update_effectiveness(params, w_before, step_id=""):
    w_after = torch.cat([p.view(-1) for p in params]).detach()
    delta = w_after - w_before
    n_weight = delta.numel()
    changed = torch.count_nonzero(delta).item()
    print(
        f"[{step_id}] Weight Update | Δmean: {delta.mean():.4e} | Δstd: {delta.std():.4e} | "
        f"Δmax: {delta.abs().max():.4e} | changed: {changed}/{n_weight}"
    )

def evaluate_model(loader):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            with torch.no_grad():
                h = forward_node1(input_ids)
            safe_sendall(b'I')
            send_tensor(h)
            send_label_tensor(labels)
            preds.extend(recv_pred_batch(len(labels)))
            labels_all.extend(labels.cpu().tolist())

        del input_ids, labels, h
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    return metric.compute(predictions=preds, references=labels_all)

# === TRAINING LOOP ===
print("\nTraining Mode:")
if config.NODE1_USE_ZO and config.NODE2_USE_BACKPROP:
    print("Hybrid Mode: Node1 = ZO, Node2 = Backprop")
elif not config.NODE1_USE_ZO and config.NODE2_USE_BACKPROP:
    print("Full Backprop Mode")
else:
    raise NotImplementedError("Unsupported mode")

optimizer = torch.optim.Adam(params, lr=config.NODE1_LR)
mu = config.ZO_MU
P = config.ZO_PERTURBATIONS
lr = config.LEARNING_RATE

try:
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}")
        epoch_losses = []

        for step, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")

            if config.NODE1_USE_ZO:
                w = params_to_vector(params)
                grad = torch.zeros_like(w)
                losses = []

                if config.DEBUG_MODE:
                    w_before = w.clone()

                for i in range(P):
                    if config.DEBUG_MODE:
                        print('P:',i)
                    u = torch.randn_like(w)
                    w_pos = w + mu * u
                    w_neg = w - mu * u

                    update_model_params(params, vector_to_params(w_pos, params))
                    h_pos = forward_node1(input_ids)

                    update_model_params(params, vector_to_params(w_neg, params))
                    h_neg = forward_node1(input_ids)

                    safe_sendall(b'Z')
                    send_tensor(h_pos)
                    send_label_tensor(labels)
                    send_tensor(h_neg)
                    send_label_tensor(labels)

                    L = recv_float()
                    grad += L * u
                    losses.append(abs(L))

                grad /= P
                avg_loss = np.mean(losses)
                epoch_losses.append(avg_loss)
                new_w = w - lr * grad
                update_model_params(params, vector_to_params(new_w, params))

                if config.DEBUG_MODE:
                    check_param_update_effectiveness(params, w_before, step_id=f"Epoch{epoch+1}-Step{step+1}-ZO")

            else:
                optimizer.zero_grad()
                hidden = forward_node1(input_ids)
                hidden.requires_grad_()

                if config.DEBUG_MODE:
                    w_before = params_to_vector(params).clone()

                safe_sendall(b'B')
                send_tensor(hidden)
                send_label_tensor(labels)

                loss_value = recv_float()
                loss_tensor = torch.tensor(loss_value, device="cuda", requires_grad=True)
                loss_tensor.backward()
                optimizer.step()
                epoch_losses.append(loss_value)

                if config.DEBUG_MODE:
                    check_param_update_effectiveness(params, w_before, step_id=f"Epoch{epoch+1}-Step{step+1}-BP")

        train_acc = evaluate_model(train_loader)["accuracy"]
        val_acc = evaluate_model(val_loader)["accuracy"]
        test_acc = evaluate_model(test_loader)["accuracy"]
        
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

except Exception as e:
    print(f"Training crashed: {e}")
finally:
    sock.close()
    print("Socket closed")

model.save_pretrained(os.path.join(run_dir, "opt-sst2-finetuned"))
tokenizer.save_pretrained(os.path.join(run_dir, "opt-sst2-finetuned"))
print(f"Model saved to {run_dir}/opt-sst2-finetuned")
print(f"Logs saved to {log_path}")
