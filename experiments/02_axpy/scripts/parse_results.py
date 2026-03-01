#!/usr/bin/env python3
# File: plot_vecadd.py
# Usage:
# python plot_vecadd.py vecadd_results.csv bandwidth_output.png flops_output.png

import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 4:
    print("Usage: python parse_results.py <csv_file> <bandwidth_output_file> <flops_output_file>")
    sys.exit(1)

csv_file = sys.argv[1]
bandwidth_output = sys.argv[2]
flops_output = sys.argv[3]

# 1️⃣ Read CSV
df = pd.read_csv(csv_file)

# Expecting column: kernel_time_ms
if 'kernel_time_ms' not in df.columns:
    print("CSV must contain column: kernel_time_ms")
    sys.exit(1)

# 2️⃣ Compute total data moved in MB
# 3 arrays (A,B,C), float = 4 bytes
df['total_data_MB'] = df['N'] * 3 * 4 / (1024*1024)

# 3️⃣ Compute FLOPS (vector add = 1 FLOP per element)
df['kernel_time_s'] = df['kernel_time_ms'] / 1000.0
df['FLOPS'] = df['N'] / df['kernel_time_s']
df['GFLOPS'] = df['FLOPS'] / 1e9

# 4️⃣ Compute averages
avg_bw = df.groupby('total_data_MB')['effective_bandwidth_GBps'].mean().reset_index()
avg_flops = df.groupby('total_data_MB')['GFLOPS'].mean().reset_index()

# ==============================
# 📈 Plot 1: Bandwidth
# ==============================
plt.figure(figsize=(10,6))

for data_MB, group in df.groupby('total_data_MB'):
    plt.scatter([data_MB]*len(group), group['effective_bandwidth_GBps'], alpha=0.3)

plt.plot(avg_bw['total_data_MB'],
         avg_bw['effective_bandwidth_GBps'],
         marker='o',
         linestyle='-',
         label='Average Bandwidth')

plt.title('Vector Add Effective Bandwidth vs Total Data Moved')
plt.xlabel('Total Data Moved (MB)')
plt.ylabel('Effective Bandwidth (GB/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(bandwidth_output)
plt.close()

print(f"Bandwidth plot saved to {bandwidth_output}")

# ==============================
# 📈 Plot 2: FLOPS
# ==============================
plt.figure(figsize=(10,6))

for data_MB, group in df.groupby('total_data_MB'):
    plt.scatter([data_MB]*len(group), group['GFLOPS'], alpha=0.3)

plt.plot(avg_flops['total_data_MB'],
         avg_flops['GFLOPS'],
         marker='o',
         linestyle='-',
         label='Average GFLOPS')

plt.title('Vector Add GFLOPS vs Total Data Moved')
plt.xlabel('Total Data Moved (MB)')
plt.ylabel('GFLOPS')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(flops_output)
plt.close()

print(f"FLOPS plot saved to {flops_output}")
print(f"Embed in Markdown:")
print(f"![Bandwidth]({bandwidth_output})")
print(f"![FLOPS]({flops_output})")