import matplotlib.pyplot as plt
import numpy as np

# 1. Data Definition
RUN_SIZE = {
    "1K":   {"Other": 8.05, "Memset": 0.07, "Memcpy": 1.49, "Kernel": 60.06, "DataLoader": 0.00, "CPU Exec": 30.33},
    "2K":   {"Other": 7.50, "Memset": 0.10, "Memcpy": 2.20, "Kernel": 62.00, "DataLoader": 0.50, "CPU Exec": 27.70},
    "4K":   {"Other": 6.80, "Memset": 0.15, "Memcpy": 3.10, "Kernel": 65.00, "DataLoader": 1.00, "CPU Exec": 23.95},
    "8K":   {"Other": 6.00, "Memset": 0.20, "Memcpy": 4.50, "Kernel": 68.00, "DataLoader": 1.50, "CPU Exec": 19.80},
    "16K":  {"Other": 5.20, "Memset": 0.25, "Memcpy": 6.00, "Kernel": 72.00, "DataLoader": 2.00, "CPU Exec": 14.55},
    "32K":  {"Other": 4.50, "Memset": 0.30, "Memcpy": 8.50, "Kernel": 75.00, "DataLoader": 2.50, "CPU Exec": 9.20},
    "64K":  {"Other": 4.00, "Memset": 0.40, "Memcpy": 11.0, "Kernel": 78.00, "DataLoader": 3.00, "CPU Exec": 3.60},
    "128K": {"Other": 3.50, "Memset": 0.50, "Memcpy": 14.0, "Kernel": 79.50, "DataLoader": 2.00, "CPU Exec": 0.50},
}

# 2. Sorting and Preparation Logic
def run_size_key(label: str) -> int:
    """Sorts '1K', '2K', etc., numerically by converting suffixes."""
    s = label.strip().upper()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    try:
        return int(s)
    except ValueError:
        return 10**12

# Get sorted labels
run_labels = sorted(RUN_SIZE.keys(), key=run_size_key)

# Define stacking order (bottom to top)
categories = ["CPU Exec", "Kernel", "Memcpy", "Memset", "DataLoader", "Other"]

# 3. Data Matrix and Normalization
vals = np.array([[RUN_SIZE[r].get(c, 0.0) for r in run_labels] for c in categories], dtype=float)

# Normalize to 100% per column
col_sums = vals.sum(axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    vals = np.where(col_sums > 0, vals * (100.0 / col_sums), vals)

# 4. Plotting
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(run_labels))
bottom = np.zeros(len(run_labels))

# Custom color palette (optional)
colors = plt.cm.get_cmap('tab10').colors

for i, c in enumerate(categories):
    label = f"{c} Time" if "Time" not in c else c
    ax.bar(x, vals[i], bottom=bottom, label=label, color=colors[i % len(colors)], width=0.7)
    
    # Optional: Percentage labels inside bars (only for segments > 4% for readability)
    for j in range(len(run_labels)):
        if vals[i][j] > 4.0:
            ax.text(j, bottom[j] + vals[i][j]/2, f'{vals[i][j]:.1f}%', 
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    
    bottom += vals[i]

# 5. Styling
ax.set_title("Percentage Breakdown of Step Time by Category (1K–128K)", fontsize=14, pad=15)
ax.set_ylabel("Percentage of Total Step Time (%)", fontsize=12)
ax.set_xlabel("Run Size", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(run_labels)
ax.set_ylim(0, 100)

# Move legend outside the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# 6. Output
plt.savefig("step_time_percentage_breakdown.png", dpi=300, bbox_inches='tight')
plt.show()