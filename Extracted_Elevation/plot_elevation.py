import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ELEVATION_FILE = BASE_DIR / "extracted_average_shape_elevation.csv"

OUTPUT_DIR = BASE_DIR / "output_average_elevation"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. READ DATA
# ============================================================

df = pd.read_csv(ELEVATION_FILE, skipinitialspace=True)

df.columns = df.columns.str.strip()

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
df["MEAN"] = pd.to_numeric(df["MEAN"], errors="coerce")

df = df.dropna(subset=["YEAR", "MEAN"])

df = df.sort_values("YEAR")


# ============================================================
# 3. LINEAR REGRESSION
# ============================================================

x = df["YEAR"].values
y = df["MEAN"].values

slope, intercept, r_value, p_value, std_err = linregress(x, y)

r_squared = r_value ** 2

# Regression line
y_pred = slope * x + intercept


# ============================================================
# 4. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))


# Observed data
ax.plot(
    x,
    y,
    marker="o",
    linewidth=1.8,
    markersize=5,
    label="Mean elevation"
)


# Regression trendline
ax.plot(
    x,
    y_pred,
    linestyle="--",
    linewidth=2,
    label=(
        f"Linear trend: {slope:.2f} m/year\n"
        f"$R^2$ = {r_squared:.3f}, "
        f"$p$ = {p_value:.4f}"
    )
)


# Labels
ax.set_xlabel("Year", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Elevation (m)", fontsize=12, fontweight="bold")

ax.set_title(
    "Mean Elevation of Accumulation Area\nJefferson Park Glacier",
    fontsize=14, fontweight="bold"
)


# X-axis
ax.set_xticks(np.arange(
    df["YEAR"].min(),
    df["YEAR"].max() + 1,
    5
))


# Grid and legend
ax.grid(True, linestyle="--", alpha=0.4)

ax.legend(fontsize=10)

plt.tight_layout()


# ============================================================
# 5. SAVE FIGURE
# ============================================================

OUTPUT_FILE = OUTPUT_DIR / "Jefferson_Park_Accumulation_Area_Mean_Elevation_Regression.png"

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 6. PRINT REGRESSION RESULTS
# ============================================================

print("\n========================================")
print("Mean Elevation Linear Regression")
print("Jefferson Park Glacier")
print("========================================")

print(f"Slope:       {slope:.4f} m/year")
print(f"Intercept:   {intercept:.4f}")
print(f"R-squared:  {r_squared:.4f}")
print(f"P-value:    {p_value:.6f}")
print(f"Std error:  {std_err:.4f} m/year")

print("========================================")

print(f"Figure saved to: {OUTPUT_FILE}")