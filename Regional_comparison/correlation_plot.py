# ============================================================
# Regional comparison: Correlation matrix of Oregon glaciers
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Load data
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "all_AAR.csv"

df = pd.read_csv(input_file)

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

df = df.sort_values("YEAR").reset_index(drop=True)

output_dir = BASE_DIR / "Correlation"

output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Read data
# ============================================================

df = pd.read_csv(input_file)

# Sort by year
df = df.sort_values("YEAR")

# ============================================================
# 3. Glacier columns
# ============================================================

glacier_columns = [
    "Hayden_AAR_interp",
    "Collier_AAR_interp",
    "Eliot_AAR_interp",
    "Reid_AAR_interp",
    "Jefferson_AAR_interp"
]

# Glacier names for plotting
glacier_names = [
    "Hayden",
    "Collier",
    "Eliot",
    "Reid",
    "Jefferson"
]

# ============================================================
# 4. Calculate Pearson correlation matrix
# ============================================================

aar_data = df[glacier_columns].copy()

# Rename columns for plotting
aar_data.columns = glacier_names

correlation_matrix = aar_data.corr(method="pearson")

# ============================================================
# 5. Plot correlation heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0,
    cbar_kws={
        "label": "Pearson correlation coefficient (r)"
    },
    annot_kws={
        "size": 14
    },
    ax=ax
)

# Title
ax.set_title(
    "Correlation between Oregon glaciers",
    fontsize=18,
    fontweight="bold",
    pad=20
)

# Axis labels
ax.set_xlabel("")
ax.set_ylabel("")

# Tick labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    fontsize=13,
    fontweight="bold",
    rotation=0
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    fontsize=13,
    fontweight="bold",
    rotation=90
)

plt.tight_layout()

# ============================================================
# 6. Save figure
# ============================================================

output_figure = output_dir / "Oregon_glacier_correlation_matrix.png"

plt.savefig(
    output_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved to: {output_figure}")

# ============================================================
# 7. Save correlation matrix
# ============================================================

output_csv = output_dir / "Oregon_glacier_correlation_matrix.csv"

correlation_matrix.to_csv(output_csv)

print(f"Correlation matrix saved to: {output_csv}")