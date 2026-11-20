import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AAR_FILE = BASE_DIR / "AAR_interpolation_extrapolation_19Nov.csv"

SNOTEL_FILE = BASE_DIR / "snotel_hogg_pass_mud_ridge.csv"

OUTPUT_DIR = BASE_DIR / "April_SWE_AAR_Regression"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. READ AAR DATA
# ============================================================

aar_data = pd.read_csv(AAR_FILE)

aar_data = aar_data.rename(columns={
    "YEAR": "Year"
})

aar_data = aar_data[["Year", "AAR"]].copy()

aar_data["Year"] = pd.to_numeric(
    aar_data["Year"],
    errors="coerce"
)

aar_data["AAR"] = pd.to_numeric(
    aar_data["AAR"],
    errors="coerce"
)

aar_data = aar_data.dropna()

print("\nAAR data:")
print(aar_data.head())


# ============================================================
# 3. READ SNOTEL APRIL SWE DATA
# ============================================================

snotel_data = pd.read_csv(SNOTEL_FILE)

# Rename columns
snotel_data = snotel_data.rename(columns={
    snotel_data.columns[0]: "Year",
    snotel_data.columns[2]: "April_SWE_in"
})

snotel_data = snotel_data[[
    "Year",
    "April_SWE_in"
]].copy()

# Convert to numeric
snotel_data["Year"] = pd.to_numeric(
    snotel_data["Year"],
    errors="coerce"
)

snotel_data["April_SWE_in"] = pd.to_numeric(
    snotel_data["April_SWE_in"],
    errors="coerce"
)

snotel_data = snotel_data.dropna()

# Convert inches to mm
snotel_data["April_SWE_mm"] = (
    snotel_data["April_SWE_in"] * 25.4
)

print("\nSNOTEL data:")
print(snotel_data.head())


# ============================================================
# 4. MERGE AAR AND APRIL SWE DATA
# ============================================================

merged = pd.merge(
    aar_data,
    snotel_data,
    on="Year",
    how="inner"
)

merged = merged.sort_values("Year")

print("\nMerged data:")
print(merged)

print(f"\nNumber of matched years: {len(merged)}")


# ============================================================
# 5. LINEAR REGRESSION
# ============================================================

x = merged["April_SWE_mm"]
y = merged["AAR"]

regression = linregress(x, y)

slope = regression.slope
intercept = regression.intercept
r_value = regression.rvalue
r_squared = regression.rvalue**2
p_value = regression.pvalue
std_err = regression.stderr

print("\n" + "="*60)
print("APRIL SWE vs AAR LINEAR REGRESSION")
print("="*60)

print(f"Number of observations (n): {len(merged)}")
print(f"Slope: {slope:.5f}")
print(f"Intercept: {intercept:.5f}")
print(f"R²: {r_squared:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Standard error: {std_err:.5f}")


# ============================================================
# 6. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))


# Scatter plot
scatter = ax.scatter(
    x,
    y,
    color="steelblue",
    edgecolor="steelblue",
    alpha=0.7,
    s=80,
    label="AAR"
)

# ------------------------------------------------------------
# Linear regression line
# ------------------------------------------------------------

x_line = np.linspace(x.min(), x.max(), 100)

y_line = slope * x_line + intercept

# Regression line
regression_line, = ax.plot(
    x_line,
    y_line,
    color="darkorange",
    linewidth=3,
    label="Linear regression"
)



# ------------------------------------------------------------
# Regression statistics
# ------------------------------------------------------------

stats_label = (
    f"P-value = {p_value:.4f}\n"
    f"R² = {r_squared:.2f}"
)

# Invisible artist for statistics in legend
stats_handle = ax.plot(
    [],
    [],
    linestyle="",
    label=stats_label
)[0]


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    handles=[scatter, regression_line, stats_handle],
    loc="lower right",
    fontsize=12
)


# ------------------------------------------------------------
# Labels and title
# ------------------------------------------------------------

ax.set_xlabel(
    "April snow water equivalent (mm)",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "AAR",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "AAR vs April SWE",
    fontsize=18,
    fontweight="bold"
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    True,
    linestyle="-",
    alpha=0.3
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.tick_params(
    axis="both",
    labelsize=12
)

plt.tight_layout()


# ============================================================
# 7. SAVE FIGURE
# ============================================================

output_file = OUTPUT_DIR / "AAR_vs_April_SWE.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nFigure saved to:\n{output_file}")


# ============================================================
# 7. SAVE FIGURE
# ============================================================

output_file = OUTPUT_DIR / "AAR_vs_April_SWE.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nFigure saved to:\n{output_file}")


# ============================================================
# 8. EXPORT MERGED DATA
# ============================================================

merged_output = OUTPUT_DIR / "AAR_April_SWE_merged.csv"

merged.to_csv(
    merged_output,
    index=False
)

print(f"Merged data saved to:\n{merged_output}")