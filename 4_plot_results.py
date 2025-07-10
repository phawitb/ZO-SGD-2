import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Create output folder
plot_dir = "output_plots"
os.makedirs(plot_dir, exist_ok=True)

# Mapping: name -> log path and output plot name
method = {
    'Full Backpropagation': {
        "log": "/home/phawit/Documents/MyDisk/Z0-SGDX-2/output/train_50/log_full_bp_train.csv",
        "out": "full_backpropagation.png"
    },
    'Full ZO-SGD': {
        "log": "/home/phawit/Documents/MyDisk/Z0-SGDX-2/output/train_26/log_full_zo_train.csv",
        "out": "full_zosgd.png"
    },
    'Split[-,Backprop]': {
        "log": "/home/phawit/Documents/MyDisk/Z0-SGDX-2/output/train_38/log_split.csv",
        "out": "split_nozo_backprop.png"
    },
    'Split[ZO,Backprop]': {
        "log": "/home/phawit/Documents/MyDisk/Z0-SGDX-2/output/train_49/log_split.csv",
        "out": "split_zo_backprop.png"
    },
}

# Plot each individual method
def plot_individual(log_path, save_name):
    df = pd.read_csv(log_path)

    plt.figure(figsize=(12, 6))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
    plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['loss'], label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_name))
    plt.close()
    print(f"[✓] Saved: {save_name}")

# Collect summary info
acc_summary = {'Method': [], 'Train': [], 'Validation': [], 'Test': [], 'Time/Epoch (s)': []}

for name, paths in method.items():
    log_path = paths["log"]
    save_name = paths["out"]
    
    # Plot and save each method
    plot_individual(log_path, save_name)

    # Read CSV for summary
    df = pd.read_csv(log_path)
    acc_summary['Method'].append(name)
    acc_summary['Train'].append(df['train_acc'].iloc[-1])
    acc_summary['Validation'].append(df['val_acc'].iloc[-1])
    acc_summary['Test'].append(df['test_acc'].iloc[-1])

    # Calculate time/epoch
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        epoch_durations = (timestamps - timestamps.shift()).dt.total_seconds().dropna()
        avg_time = epoch_durations.mean()
        acc_summary['Time/Epoch (s)'].append(round(avg_time, 2))
    else:
        acc_summary['Time/Epoch (s)'].append(None)

# Create summary DataFrame
summary_df = pd.DataFrame(acc_summary)

# === Plot summary bar chart ===
x = range(len(summary_df))
bar_width = 0.2

plt.figure(figsize=(10, 6))
plt.bar([i - bar_width for i in x], summary_df['Train'], width=bar_width, label='Train Acc')
plt.bar(x, summary_df['Validation'], width=bar_width, label='Val Acc')
plt.bar([i + bar_width for i in x], summary_df['Test'], width=bar_width, label='Test Acc')

plt.xticks(x, summary_df['Method'], rotation=0)
plt.ylabel('Accuracy')
plt.title('Final Accuracy Comparison')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "summary_acc.png"))
plt.close()
print("Saved: summary_acc.png")

# Print and save summary
print("\n=== Accuracy and Time Summary ===")
print(summary_df)

summary_df.to_csv(os.path.join(plot_dir, "summary_table.csv"), index=False)
print("Saved: summary_table.csv")
