import socket
import struct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import OPTForSequenceClassification
import config
import random

# === Set Seed ===
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Setup ===
HOST = "0.0.0.0"
PORT = config.NODE2_PORT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model and extract layers ===
model = OPTForSequenceClassification.from_pretrained(config.MODEL_PATH).to(device)
model.train()
split_idx = config.SPLIT_LAYER_IDX
later_layers = model.model.decoder.layers[split_idx:]
final_norm = model.model.decoder.final_layer_norm
classifier_head = model.score

# === Optimizer ===
params = list(p for l in later_layers for p in l.parameters()) + \
         list(final_norm.parameters()) + list(classifier_head.parameters())
optimizer = torch.optim.Adam(params, lr=config.NODE2_LR)

hidden_size = model.config.hidden_size
seq_len = 128

def check_param_update_effectiveness(params, w_before, label=""):
    w_after = torch.cat([p.data.view(-1) for p in params]).detach()
    delta = w_after - w_before
    changed = torch.count_nonzero(delta).item()
    total = delta.numel()
    print(f"{label} Weight Δ | mean: {delta.mean():.4e}, std: {delta.std():.4e}, max: {delta.abs().max():.4e}, changed: {changed}/{total}")

# === Socket Utils ===
def recv_tensor(conn, dtype, shape=None):
    try:
        length_bytes = conn.recv(4)
        if not length_bytes:
            raise ConnectionError("Missing tensor header")
        length = struct.unpack('!I', length_bytes)[0]

        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                raise ConnectionError("Tensor data incomplete")
            data += packet

        array = np.frombuffer(data, dtype=dtype).copy()
        tensor = torch.tensor(array, device=device)
        if shape and tensor.numel() != np.prod(shape):
            raise ValueError(f"Received tensor size mismatch: expected {np.prod(shape)}, got {tensor.numel()}")
        return tensor.view(*shape) if shape else tensor

    except Exception as e:
        raise RuntimeError(f"recv_tensor failed: {e}")

def recv_hidden(conn, batch_size=None):
    array = recv_tensor(conn, np.float32)
    if batch_size is None:
        total = array.numel()
        batch_size = total // (seq_len * hidden_size)
    return array.contiguous().view(batch_size, seq_len, hidden_size)

def recv_labels(conn, batch_size):
    return recv_tensor(conn, np.int64, (batch_size,))

def send_float(conn, value):
    conn.sendall(struct.pack('!d', value))

def send_predictions(conn, preds):
    data = struct.pack(f'!{len(preds)}I', *preds)
    conn.sendall(data)

# === Forward ===
def node2_forward(hidden):
    for layer in later_layers:
        hidden = layer(hidden)[0]
    hidden = final_norm(hidden)
    pooled = hidden[:, -1, :]
    logits = classifier_head(pooled)
    return logits

# === Start Server ===
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Node2 server listening at port {PORT}...")

while True:
    try:
        conn, addr = server.accept()
        conn.settimeout(10)
        print(f"Connected from {addr}")

        while True:
            try:
                mode = conn.recv(1)
                if not mode or mode not in [b'I', b'Z', b'B']:
                    print(f"Unknown mode received: {mode}")
                    continue

                if mode == b'I':
                    hidden = recv_hidden(conn)
                    batch_size = hidden.shape[0]
                    labels = recv_labels(conn, batch_size)

                    if torch.isnan(hidden).any() or torch.isinf(hidden).any():
                        print("NaN/Inf in hidden during inference. Skipping.")
                        continue
                    if torch.any((labels < 0) | (labels >= model.config.num_labels)):
                        print("Invalid labels during inference. Skipping.")
                        continue

                    with torch.no_grad():
                        logits = node2_forward(hidden)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("NaN/Inf in logits during inference. Skipping.")
                        continue

                    preds = torch.argmax(logits, dim=1).tolist()
                    send_predictions(conn, preds)
                    print(f"Inference: batch={batch_size}")

                elif mode == b'Z':
                    try:
                        hidden_pos = recv_hidden(conn)
                        batch_size = hidden_pos.shape[0]
                        labels_pos = recv_labels(conn, batch_size)
                        hidden_neg = recv_hidden(conn, batch_size)
                        labels_neg = recv_labels(conn, batch_size)

                        if torch.isnan(hidden_pos).any() or torch.isinf(hidden_pos).any() or \
                           torch.isnan(hidden_neg).any() or torch.isinf(hidden_neg).any():
                            print("NaN/Inf in hidden vectors (ZO mode). Skipping batch.")
                            continue

                        if torch.any((labels_pos < 0) | (labels_pos >= model.config.num_labels)) or \
                           torch.any((labels_neg < 0) | (labels_neg >= model.config.num_labels)):
                            print("Invalid labels in ZO mode. Skipping batch.")
                            continue

                        if config.DEBUG_MODE:
                            w_before = torch.cat([p.data.view(-1) for p in params]).clone()

                        optimizer.zero_grad()
                        logits_pos = node2_forward(hidden_pos)
                        logits_neg = node2_forward(hidden_neg)

                        if torch.isnan(logits_pos).any() or torch.isinf(logits_pos).any() or \
                           torch.isnan(logits_neg).any() or torch.isinf(logits_neg).any():
                            print("NaN/Inf in logits (ZO mode). Skipping batch.")
                            continue

                        L_pos = F.cross_entropy(logits_pos, labels_pos)
                        L_neg = F.cross_entropy(logits_neg, labels_neg)
                        avg_loss = (L_pos + L_neg) / 2

                        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
                            print("⚠️ NaN/Inf in ZO loss. Skipping batch.")
                            continue

                        avg_loss.backward()
                        optimizer.step()

                        if config.DEBUG_MODE:
                            check_param_update_effectiveness(params, w_before, label="ZO")

                        send_float(conn, avg_loss.item())
                        print(f"ZO-BP: batch={batch_size}, loss={avg_loss.item():.4f}")

                    except Exception as e:
                        print(f"Exception in ZO-BP: {e}")
                        continue

                elif mode == b'B':
                    hidden = recv_hidden(conn)
                    batch_size = hidden.shape[0]
                    labels = recv_labels(conn, batch_size)

                    if torch.isnan(hidden).any() or torch.isinf(hidden).any():
                        print("NaN/Inf in hidden during training. Skipping.")
                        continue
                    if torch.any((labels < 0) | (labels >= model.config.num_labels)):
                        print("Invalid labels during training. Skipping.")
                        continue

                    hidden.requires_grad_()
                    if config.DEBUG_MODE:
                        w_before = torch.cat([p.data.view(-1) for p in params]).clone()

                    optimizer.zero_grad()
                    logits = node2_forward(hidden)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("NaN/Inf in logits during training. Skipping.")
                        continue

                    loss = F.cross_entropy(logits, labels)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("Loss is NaN/Inf. Skipping.")
                        continue

                    loss.backward()
                    optimizer.step()

                    if config.DEBUG_MODE:
                        check_param_update_effectiveness(params, w_before, label="BP")

                    send_float(conn, loss.item())
                    print(f"Full BP: batch={batch_size}, loss={loss.item():.4f}")

            except Exception as e:
                print(f"Session error: {e}")
                break

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        try:
            conn.close()
        except:
            pass
        print("Disconnected. Waiting for new connection...")
