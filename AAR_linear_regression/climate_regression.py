# ============================================================
# Mt. Jefferson Glacier
# Preliminary Climate–AAR Linear Regression Analysis
#
# Input:
#   AAR_ver0.csv
#   PRISM_ppt_tmean_Preliminary_800m_197901_202610_44.6841_-121.8070.csv
#
# Output:
#   Processed climate data
#   Matched AAR-climate data
#   Climate time-series plots
#   AAR-climate scatter plots with regression lines
#   Regression statistics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AAR_FILE = BASE_DIR / "AAR_ver0.csv"

PRISM_FILE = BASE_DIR / (
    "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6828_-121.8067.csv"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. READ PRISM DATA
# ============================================================

# Skip the metadata at the beginning of the PRISM file
climate = pd.read_csv(PRISM_FILE, skiprows=11)

# Rename columns
climate.columns = [
    "Date",
    "Precipitation",
    "Tmin",
    "Tmean",
    "Tmax"
]

# Convert date
climate["Date"] = pd.to_datetime(
    climate["Date"],
    format="%Y-%m"
)

# Convert temperature variables to numeric
for col in ["Tmin", "Tmean", "Tmax"]:
    climate[col] = pd.to_numeric(
        climate[col],
        errors="coerce"
    )

# Extract year and month
climate["Year"] = climate["Date"].dt.year
climate["Month"] = climate["Date"].dt.month

# Use Tmean as the temperature variable
climate["Temperature"] = climate["Tmean"]

print("\n===== PRISM CLIMATE DATA =====")
print(climate.head())

print(
    "Climate period:",
    climate["Date"].min(),
    "to",
    climate["Date"].max()
)


# ============================================================
# 3. CALCULATE ANNUAL CLIMATE VARIABLES
# ============================================================

annual_climate = climate.groupby("Year").agg(
    Annual_Temperature=("Temperature", "mean"),
    Annual_Precipitation=("Precipitation", "sum")
)

# Annual temperature = mean of monthly mean temperatures
# Annual precipitation = sum of monthly precipitation


# ============================================================
# 4. CALCULATE SEASONAL CLIMATE VARIABLES
# ============================================================

# Summer: June, July, August
summer = climate[climate["Month"].isin([6, 7, 8])]

summer_climate = summer.groupby("Year").agg(
    Summer_Temperature=("Temperature", "mean"),
    Summer_Precipitation=("Precipitation", "sum")
)

# Winter: December, January, February
# Assign December to the following winter year.
winter = climate.copy()

winter["Winter_Year"] = winter["Year"]

winter.loc[
    winter["Month"] == 12,
    "Winter_Year"
] += 1

winter = winter[
    winter["Month"].isin([12, 1, 2])
]

winter_climate = winter.groupby("Winter_Year").agg(
    Winter_Temperature=("Temperature", "mean"),
    Winter_Precipitation=("Precipitation", "sum")
)

winter_climate.index.name = "Year"


# ============================================================
# 5. COMBINE CLIMATE VARIABLES
# ============================================================

climate_summary = annual_climate.join(
    summer_climate,
    how="outer"
)

climate_summary = climate_summary.join(
    winter_climate,
    how="outer"
)

climate_summary = climate_summary.sort_index()

# Export processed climate data
climate_summary.to_csv(
    OUTPUT_DIR / "Climate_summary.csv"
)

print("\n===== CLIMATE SUMMARY =====")
print(climate_summary.head())


# ============================================================
# 6. READ AAR DATA
# ============================================================

aar = pd.read_csv(AAR_FILE)

# Clean column names
aar.columns = aar.columns.str.strip()

# Convert numeric columns
aar["Year"] = pd.to_numeric(
    aar["Year"],
    errors="coerce"
)

aar["AAR"] = pd.to_numeric(
    aar["AAR"],
    errors="coerce"
)

aar = aar.dropna(subset=["Year", "AAR"])

aar["Year"] = aar["Year"].astype(int)

# Sort chronologically
aar = aar.sort_values("Year")

print("\n===== AAR DATA =====")
print(aar.head())
print(aar.tail())

print("\nMissing AAR years:")
expected_years = set(range(1985, 2027))
available_years = set(aar["Year"])

print(sorted(expected_years - available_years))


# ============================================================
# 7. MATCH AAR WITH CLIMATE DATA
# ============================================================

# Merge by year
merged = aar.merge(
    climate_summary,
    left_on="Year",
    right_index=True,
    how="left"
)

# Remove rows where the relevant climate variable is missing
# Missing AAR years are already absent from the AAR dataset.

merged = merged.sort_values("Year")

# Export matched dataset
merged.to_csv(
    OUTPUT_DIR / "AAR_climate_matched.csv",
    index=False
)

print("\n===== MATCHED DATA =====")
print(merged.head())

print("\nMatched data shape:")
print(merged.shape)


# ============================================================
# 8. PLOT CLIMATE TIME SERIES
# ============================================================

# Annual temperature
plt.figure(figsize=(12, 5))

plt.plot(
    climate_summary.index,
    climate_summary["Annual_Temperature"],
    linewidth=1.5
)

plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.title("Annual Mean Temperature - PRISM")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Annual_Temperature.png",
    dpi=300
)

plt.show()


# Annual precipitation
plt.figure(figsize=(12, 5))

plt.plot(
    climate_summary.index,
    climate_summary["Annual_Precipitation"],
    linewidth=1.5
)

plt.xlabel("Year")
plt.ylabel("Precipitation (mm)")
plt.title("Annual Precipitation - PRISM")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Annual_Precipitation.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. PLOT AAR TIME SERIES
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    aar["Year"],
    aar["AAR"],
    marker="o",
    linewidth=1.5
)

plt.xlabel("Year")
plt.ylabel("AAR")
plt.title("Accumulation Area Ratio - Jefferson Park Glacier")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "AAR_timeseries.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. LINEAR REGRESSION FUNCTION
# ============================================================

def perform_regression(
    data,
    x_column,
    y_column="AAR",
    x_label=None,
    y_label="AAR",
    output_name=None
):

    # Remove missing values
    valid = data[[x_column, y_column]].dropna()

    x = valid[x_column]
    y = valid[y_column]

    # Linear regression
    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    p_value = result.pvalue
    stderr = result.stderr

    # Regression equation
    equation = f"y = {slope:.4f}x + {intercept:.4f}"

    print("\n===================================")
    print(f"Regression: {y_column} vs {x_column}")
    print("===================================")

    print(f"Number of observations: {len(valid)}")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R²: {r_squared:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Standard error of slope: {stderr:.6f}")
    print(f"Equation: {equation}")

    # Plot
    plt.figure(figsize=(8, 6))

    plt.scatter(
        x,
        y,
        s=40,
        alpha=0.8
    )

    # Regression line
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    plt.plot(
        x_fit,
        y_fit,
        linewidth=2,
        label="Linear regression"
    )

    plt.xlabel(x_label if x_label else x_column)
    plt.ylabel(y_label)

    plt.title(
        f"{y_column} vs {x_column}\n"
        f"R² = {r_squared:.3f}, p = {p_value:.3f}"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_name:
        plt.savefig(
            OUTPUT_DIR / f"{output_name}.png",
            dpi=300
        )

    plt.show()

    return {
        "Variable": x_column,
        "N": len(valid),
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_squared,
        "p_value": p_value,
        "Slope_SE": stderr
    }


# ============================================================
# 11. PERFORM LINEAR REGRESSIONS
# ============================================================

regression_results = []

# AAR vs annual temperature
regression_results.append(
    perform_regression(
        merged,
        x_column="Annual_Temperature",
        x_label="Annual Mean Temperature (°C)",
        output_name="AAR_vs_Annual_Temperature"
    )
)

# AAR vs summer temperature
regression_results.append(
    perform_regression(
        merged,
        x_column="Summer_Temperature",
        x_label="Summer Mean Temperature (°C)",
        output_name="AAR_vs_Summer_Temperature"
    )
)

# AAR vs annual precipitation
regression_results.append(
    perform_regression(
        merged,
        x_column="Annual_Precipitation",
        x_label="Annual Precipitation (mm)",
        output_name="AAR_vs_Annual_Precipitation"
    )
)

# AAR vs winter precipitation
regression_results.append(
    perform_regression(
        merged,
        x_column="Winter_Precipitation",
        x_label="Winter Precipitation (mm)",
        output_name="AAR_vs_Winter_Precipitation"
    )
)


# ============================================================
# 12. EXPORT REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame(
    regression_results
)

regression_results.to_csv(
    OUTPUT_DIR / "Regression_results.csv",
    index=False
)

print("\n===== FINAL REGRESSION RESULTS =====")
print(regression_results)


# ============================================================
# 13. END
# ============================================================

print("\nAnalysis complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")