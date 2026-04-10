import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==============================
# CONFIGURATION
# ==============================

csv_files = [
    "exp1_baseline_globaltotal_mastered.csv",
    "exp1_hier_globaltotal_mastered.csv"
]

labels = [
    "Baseline PPO",
    "Hierarchical PPO"
]

title = "Global Mastery Comparison"
y_label = "Total Mastered Items"

output_file = "fig_5_4_global_masterypng"


# ==============================
# STYLE
# ==============================

sns.set_style("whitegrid")

plt.rcParams.update({
    "figure.figsize": (9,5),
    "figure.facecolor": "#e9e9e9",
    "axes.facecolor": "#e9e9e9",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "font.family": "serif",
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13
})


# ==============================
# PLOT
# ==============================

fig, ax = plt.subplots()

for csv, label in zip(csv_files, labels):

    df = pd.read_csv(csv).sort_values("Step")

    steps = df["Step"] / 1_000_000
    values = df["Value"]

    ax.plot(steps, values, linewidth=4, label=label)


ax.set_title(title)
ax.set_xlabel("Training Steps (Millions)")
ax.set_ylabel(y_label)

if len(labels) > 1:
    ax.legend()

for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()