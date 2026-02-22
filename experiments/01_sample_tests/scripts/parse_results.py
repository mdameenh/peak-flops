#!/usr/bin/env python3
# File: plot_vecadd.py
# Usage: python plot_vecadd.py vecadd_results.csv <output_file.png>

import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Usage: python plot_vecadd.py <csv_file> <output_file>")
    sys.exit(1)

csv_file = sys.argv[1]
output_file = sys.argv[2]

# 1️⃣ Read CSV
df = pd.read_csv(csv_file)

# 2️⃣ Compute total data moved in MB for each N
# 3 arrays (A,B,C), float=4 bytes
df['total_data_MB'] = df['N'] * 3 * 4 / (1024*1024)

# 3️⃣ Compute average bandwidth per total_data_MB
avg_df = df.groupby('total_data_MB')['effective_bandwidth_GBps'].mean().reset_index()

# 4️⃣ Plot
plt.figure(figsize=(10,6))

# Plot all individual runs as semi-transparent points
for data_MB, group in df.groupby('total_data_MB'):
    plt.scatter([data_MB]*len(group), group['effective_bandwidth_GBps'], color='blue', alpha=0.3)

# Plot average bandwidth as a line
plt.plot(avg_df['total_data_MB'], avg_df['effective_bandwidth_GBps'], color='red', marker='o', linestyle='-', label='Average Bandwidth')

plt.title('Vector Add Effective Bandwidth vs Total Data Moved')
plt.xlabel('Total Data Moved (MB)')
plt.ylabel('Effective Bandwidth (GB/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 5️⃣ Save PNG
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
print(f"Embed in Markdown: ![Vector Add Bandwidth]({output_file})")