import numpy as np
import pandas as pd

from data_processing.dataset import OHLCDataset
from config.env_loader import get_config

import matplotlib.pyplot as plt


_cfg = get_config()
sliding_step = _cfg.sliding_step
seq_len = _cfg.seq_len
csv_path = _cfg.csv_path

dataset = OHLCDataset(device='cpu')

# Instead of dataset.shape:
print(f"Number of samples: {len(dataset)}")
print(f"Sequence shape (X): {dataset.dataset.shape}")  # [Num_Samples, Seq_Len, Features]
print(f"Reference shape (y): {dataset.reference_k.shape}")  # [Num_Samples, Seq_Len, Look_Ahead]


# check if calculated close gaps can match close prices
dataset_size = len(dataset)
indices = list(range(dataset_size))
index = np.random.choice(indices, size=1)[0]
print(f"Sample index: {index}")
x, y = dataset[index]
x = x.squeeze(0)
y = y.squeeze(0)


print("--- First Sample Window ---")
print(f"X (Features) Shape: {x.shape}")
print(f"y (Look-ahead) Shape: {y.shape}")

print("\nFirst time-step features:")
print(f"x : {x[0].tolist()}")
print(f"y : {y.tolist()}")



df = pd.read_csv(csv_path)




combined_data = x[:, 3].tolist() + y[-1, :].tolist()
combined_data = np.cumsum(combined_data)
plt.figure(figsize=(10, 4))

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(combined_data, label='Close Gap', marker='x', linestyle='-', linewidth='1', color='blue')
ax2 = ax1.twinx()
ax2.plot(df['close'].values[index * sliding_step: index * sliding_step + seq_len + 10], label='Close Price', marker='o', linestyle='dotted',
         linewidth='1', color='red')
fig.tight_layout()
plt.show()