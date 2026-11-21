import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ELEVATION_FILE = BASE_DIR / "extracted_average_shape_elevation_19Nov.csv"

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
# Split the line at the missing year 2012.

BLUE = "#1f77b4"

# Line before missing year
df_before = df[df["YEAR"] <= 2011]

ax.plot(
    df_before["YEAR"],
    df_before["MEAN"],
    color=BLUE,
    marker="o",
    linewidth=1.8,
    markersize=5,
    label="Mean elevation"
)

# Line after missing year
df_after = df[df["YEAR"] >= 2013]

ax.plot(
    df_after["YEAR"],
    df_after["MEAN"],
    color=BLUE,
    marker="o",
    linewidth=1.8,
    markersize=5
)

# Dashed connection across missing year 2012
before = df.loc[df["YEAR"] == 2011]
after = df.loc[df["YEAR"] == 2013]

if not before.empty and not after.empty:
    ax.plot(
        [2011, 2013],
        [
            before["MEAN"].iloc[0],
            after["MEAN"].iloc[0]
        ],
        color="#8C8177",
        linestyle="--",
        linewidth=1.6,
        alpha=0.95,
        zorder=3,
        label="Connection across missing year (2012)"
    )


# Regression trendline
ax.plot(
    x,
    y_pred,
    linestyle="--",
    color="#FF7F0E",
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

OUTPUT_FILE = OUTPUT_DIR / "Jefferson_Park_Accumulation_Area_Mean_Elevation_Regression_19Nov.png"

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