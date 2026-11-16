# ============================================================
# Mt. Jefferson Glacier
# AAR vs Seasonal / Monthly-Combination Temperature Regression
#
# Temperature combinations:
#   June-August (JJA)
#   June-September (JJAS)
#   June-October (JJASO)
#   May-October (MJJASO)
#   May-September (MJJAS)
#   May-August (MJJA)
#
# For each year:
#   1. Calculate the mean temperature across the selected months.
#   2. Calculate temperature anomaly relative to the mean of the
#      corresponding seasonal temperature series over 1985-2026.
#   3. Regress AAR against that temperature anomaly.
#
# Outputs are saved in a separate folder:
#   output_month_combinations/
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AAR_FILE = BASE_DIR / "AAR_interpolation_extrapolation_16Nov.csv"
PRISM_FILE = BASE_DIR / (
    "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv"
)

OUTPUT_DIR = BASE_DIR / "output_month_combinations"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_START = 1991
BASELINE_END = 2020


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
# 3. READ AAR DATA
# ============================================================

aar = pd.read_csv(AAR_FILE)
aar.columns = aar.columns.str.strip()

aar["YEAR"] = pd.to_numeric(aar["YEAR"], errors="coerce")
aar["AAR"] = pd.to_numeric(aar["AAR"], errors="coerce")

aar = aar.dropna(subset=["YEAR", "AAR"])
aar["Year"] = aar["YEAR"].astype(int)
aar = (
    aar[["Year", "AAR"]]
    .sort_values("Year")
    .drop_duplicates("Year", keep="last")
)

print("\n===== AAR DATA =====")
print(aar.head())
print("Missing AAR years:", sorted(set(range(1985, 2027)) - set(aar["Year"])))


# ============================================================
# 4. DEFINE MONTH COMBINATIONS
# ============================================================

month_combinations = {
    "June-August": [6, 7, 8],
    "June-September": [6, 7, 8, 9],
    "June-October": [6, 7, 8, 9, 10],
    "May-October": [5, 6, 7, 8, 9, 10],
    "May-September": [5, 6, 7, 8, 9],
    "May-August": [5, 6, 7, 8],
    "July-August": [7, 8],
    "July-September": [7, 8, 9],
    "July-October": [7, 8, 9, 10],
    "August-October": [8, 9, 10],
    "September-October": [9, 10],
    "September-November": [9, 10, 11],
    "May-November": [5, 6, 7, 8, 9, 10, 11],
}


# ============================================================
# 5. CALCULATE ANNUAL MEAN TEMPERATURE FOR EACH COMBINATION
# ============================================================

seasonal_temperature = pd.DataFrame(index=range(1985, 2027))
seasonal_temperature.index.name = "Year"

for combination_name, months in month_combinations.items():

    selected = climate[climate["Month"].isin(months)]

    # Mean temperature for each year and selected month combination
    seasonal_mean = selected.groupby("Year")["Temperature"].agg(
        ["mean", "count"]
    )

    # Require all selected months to be available
    seasonal_mean.loc[
        seasonal_mean["count"] != len(months), "mean"
    ] = np.nan

    seasonal_temperature[combination_name] = seasonal_mean["mean"]

    # Temperature anomaly relative to the baseline period
    baseline_mean = seasonal_temperature.loc[
        BASELINE_START:BASELINE_END, combination_name
    ].mean()

    seasonal_temperature[
        f"{combination_name}_Anomaly"
    ] = seasonal_temperature[combination_name] - baseline_mean

    print(
        f"{combination_name}: baseline mean = "
        f"{baseline_mean:.3f} °C"
    )

seasonal_temperature.to_csv(
    OUTPUT_DIR / "Seasonal_Temperature_and_Anomalies.csv"
)


# ============================================================
# 6. MATCH AAR WITH SEASONAL TEMPERATURE
# ============================================================

merged = aar.merge(
    seasonal_temperature,
    left_on="Year",
    right_index=True,
    how="left"
).sort_values("Year")

merged.to_csv(
    OUTPUT_DIR / "AAR_Seasonal_Temperature_Matched.csv",
    index=False
)

print("\n===== MATCHED DATA =====")
print(merged.head())


# ============================================================
# 7. REGRESSION FUNCTION
# ============================================================

def perform_regression(
    data,
    x_column,
    plot_title,
    output_name,
    point_color="blue"
):

    valid = data[["AAR", x_column]].dropna()

    if len(valid) < 3:
        print(f"Skipping {x_column}: fewer than 3 valid observations.")
        return None

    x = valid[x_column]
    y = valid["AAR"]

    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    p_value = result.pvalue
    stderr = result.stderr

    print("\n===================================")
    print(f"Regression: AAR vs {x_column}")
    print("===================================")
    print(f"N: {len(valid)}")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R²: {r_squared:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Standard error of slope: {stderr:.6f}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        x, y,
        s=45,
        alpha=0.85,
        color=point_color,
        label="AAR"
    )

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    ax.plot(
        x_fit, y_fit,
        color="green",
        linewidth=2.5,
        label="Linear regression"
    )

    ax.set_xlabel(
        "Temperature anomaly (°C)",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_ylabel(
        "AAR",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_title(
        plot_title,
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    # P-value and R² in legend, on separate lines
    stats_label = (
        f"P-value = {p_value:.4f}\n"
        f"R² = {r_squared:.2f}"
    )

    ax.plot(
        [], [],
        linestyle="none",
        label=stats_label
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{output_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    return {
        "Combination": plot_title,
        "Temperature_Anomaly": x_column,
        "N": len(valid),
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_squared,
        "p_value": p_value,
        "Slope_SE": stderr
    }


# ============================================================
# 8. RUN ALL 12 REGRESSIONS
# ============================================================

regression_results = []

# Provide one color for each of the 12 combinations
colors = [
    "crimson", "blue", "darkorange", "purple", "teal", "magenta",
    "royalblue", "forestgreen", "sienna", "deeppink", "slateblue", "olive", "navy"
]

# Iterate over ALL combinations (12 plots)
for (combination_name, months), color in zip(
    month_combinations.items(), colors
):

    anomaly_column = f"{combination_name}_Anomaly"

    result = perform_regression(
        data=merged,
        x_column=anomaly_column,
        plot_title=f"AAR vs {combination_name} Temperature Average",
        output_name=f"AAR_vs_{combination_name.replace('-', '_')}",
        point_color=color
    )

    if result is not None:
        regression_results.append(result)


# ============================================================
# 9. EXPORT REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame(regression_results)

regression_results.to_csv(
    OUTPUT_DIR / "Regression_results_month_combinations.csv",
    index=False
)

print("\n===== FINAL REGRESSION RESULTS =====")
print(regression_results)

print("\nAnalysis complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")
