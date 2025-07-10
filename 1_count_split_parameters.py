from transformers import OPTForSequenceClassification, AutoTokenizer
import torch

# === Load model and tokenizer ===
model = OPTForSequenceClassification.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model.eval()

# === Components ===
layers = model.model.decoder.layers
embed = model.model.decoder.embed_tokens
pos_emb = model.model.decoder.embed_positions
final_norm = model.model.decoder.final_layer_norm
classifier_head = model.score

num_layers = len(layers)
hidden_size = model.config.hidden_size
seq_len = 16  # dummy input length
batch_size = 2
vocab_size = model.config.vocab_size
num_labels = model.config.num_labels

# === Dummy input ===
dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

# === Common forward through embedding ===
with torch.no_grad():
    input_embeds = embed(dummy_input_ids) + pos_emb(position_ids)

# === Get number of params for embedding ===
embed_params = list(embed.parameters()) + list(pos_emb.parameters())
embed_param_count = sum(p.numel() for p in embed_params)

# === Print header ===
print(f"{'Split':<7} {'Node1 Params':>15} {'Node2 Params':>15} {'Total':>12}  | Shapes")
print("-" * 70)

# === Loop through splits ===
for split_idx in range(1, num_layers):
    # Forward Node1
    hidden = input_embeds.clone()
    for i in range(split_idx):
        hidden = layers[i](hidden)[0]
    node1_out_shape = hidden.shape

    # Forward Node2
    node2_input = hidden.clone()
    for i in range(split_idx, num_layers):
        node2_input = layers[i](node2_input)[0]
    node2_output = final_norm(node2_input)
    pooled = node2_output[:, -1, :]  # last token
    logits = classifier_head(pooled)
    node2_out_shape = logits.shape

    # Count params
    node1_layers = layers[:split_idx]
    node2_layers = layers[split_idx:]

    node1_param_count = embed_param_count + sum(p.numel() for l in node1_layers for p in l.parameters())
    node2_param_count = sum(p.numel() for l in node2_layers for p in l.parameters())
    node2_param_count += sum(p.numel() for p in final_norm.parameters())
    node2_param_count += sum(p.numel() for p in classifier_head.parameters())

    total_param = node1_param_count + node2_param_count

    print(f"{split_idx:<7} {node1_param_count:15,} {node2_param_count:15,} {total_param:12,}  | "
          f"Node1 out: {tuple(node1_out_shape)}, Node2 out: {tuple(node2_out_shape)}")
