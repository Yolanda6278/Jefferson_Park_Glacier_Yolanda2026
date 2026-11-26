# ============================================================
# BCSD5 Climate–AAR Regression
# Regional glacier comparison (5 glaciers)
#
# BCSD5 input:
#   tas_SpatialStat_mean.csv
#   Format: Year, Month, 231 model/projection values (233 columns total)
#
# Climate predictors:
#   1. May-October mean temperature anomaly
#   2. November-October mean temperature anomaly
#   3. Water-year mean temperature anomaly (October-September)
#
# Ensemble mean is calculated across the 231 model columns for each month.
# Temperature anomalies use the 1991-2020 baseline.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

# ============================================================
# 1. FILE PATHS / SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# BCSD5 temperature SpatialStat mean CSV (no header; 233 columns)
TEMP_FILE = BASE_DIR / "bcsd5_regional" / "bcsd5" / "tas_SpatialStat_mean.csv"

AAR_FILE = BASE_DIR / "all_AAR.csv"


OUTPUT_DIR = BASE_DIR / "output_BCSD5_regression"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_START = 1991
BASELINE_END = 2020

N_MODEL_COLUMNS = 231
EXPECTED_TOTAL_COLUMNS = 233

# ============================================================
# 2. READ BCSD5 TEMPERATURE CSV
# ============================================================

def read_bcsd5_temperature(filepath):
    """
    Reads the BCSD5 tas_SpatialStat_mean.csv file.

    Expected format:
        Column 1 = Year
        Column 2 = Month
        Columns 3-233 = 231 model/projection values

    The file has no header row.
    """
    df = pd.read_csv(filepath, header=None)

    if df.shape[1] != EXPECTED_TOTAL_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_COLUMNS} columns "
            f"(Year, Month + {N_MODEL_COLUMNS} model columns), "
            f"but found {df.shape[1]} columns in {filepath}."
        )

    df = df.rename(columns={0: "Year", 1: "Month"})

    model_columns = df.columns[2:]

    # Convert all data columns to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df[model_columns] = df[model_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna(subset=["Year", "Month"])
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    # Calculate ensemble statistics across the 231 model/projection columns
    df["Temperature"] = df[model_columns].mean(axis=1, skipna=True)
    df["Temperature_Ensemble_SD"] = df[model_columns].std(axis=1, skipna=True)
    df["Temperature_N_Members"] = df[model_columns].count(axis=1)

    # Keep valid months and sort chronologically
    df = df[(df["Month"] >= 1) & (df["Month"] <= 12)].copy()
    df = df.sort_values(["Year", "Month"])
    df = df.drop_duplicates(["Year", "Month"])

    return df[
        [
            "Year",
            "Month",
            "Temperature",
            "Temperature_Ensemble_SD",
            "Temperature_N_Members",
        ]
    ]


climate = read_bcsd5_temperature(TEMP_FILE)

print("\n===== BCSD5 MONTHLY TEMPERATURE =====")
print(climate.head())
print("Climate coverage:", climate["Year"].min(), "-", climate["Year"].max())
print("Number of monthly records:", len(climate))

# Save monthly ensemble-mean climate data
climate.to_csv(OUTPUT_DIR / "BCSD5_monthly_temperature_ensemble.csv", index=False)

# ============================================================
# 3. CALCULATE CLIMATE PERIODS
# ============================================================

# Calendar-year mean temperature (January-December)
calendar = climate.groupby("Year").agg(
    Calendar_Temperature=("Temperature", "mean"),
    Calendar_Temperature_Months=("Temperature", "count"),
)

calendar.loc[
    calendar["Calendar_Temperature_Months"] != 12,
    "Calendar_Temperature"
] = np.nan

# Water year: October-September, labelled by ending year
wy = climate.copy()
wy["Water_Year"] = wy["Year"]
wy.loc[wy["Month"] >= 10, "Water_Year"] += 1

water_year = wy.groupby("Water_Year").agg(
    Water_Year_Temperature=("Temperature", "mean"),
    Water_Year_Temperature_Months=("Temperature", "count"),
)

water_year.loc[
    water_year["Water_Year_Temperature_Months"] != 12,
    "Water_Year_Temperature"
] = np.nan
water_year.index.name = "Year"

# November-October, labelled by ending year
nov_oct = climate.copy()
nov_oct["Nov_Oct_Year"] = nov_oct["Year"]
nov_oct.loc[nov_oct["Month"] >= 11, "Nov_Oct_Year"] += 1

nov_oct_period = nov_oct.groupby("Nov_Oct_Year").agg(
    Nov_Oct_Temperature=("Temperature", "mean"),
    Nov_Oct_Temperature_Months=("Temperature", "count"),
)

nov_oct_period.loc[
    nov_oct_period["Nov_Oct_Temperature_Months"] != 12,
    "Nov_Oct_Temperature"
] = np.nan
nov_oct_period.index.name = "Year"

# May-October temperature (six months, same calendar year)
may_oct = climate[
    climate["Month"].isin([5, 6, 7, 8, 9, 10])
].groupby("Year").agg(
    May_Oct_Temperature=("Temperature", "mean"),
    May_Oct_Temperature_Months=("Temperature", "count"),
)

may_oct.loc[
    may_oct["May_Oct_Temperature_Months"] != 6,
    "May_Oct_Temperature"
] = np.nan

climate_summary = (
    calendar
    .join(water_year, how="outer")
    .join(nov_oct_period, how="outer")
    .join(may_oct, how="outer")
    .sort_index()
)

# ============================================================
# 4. TEMPERATURE ANOMALIES
#    Common baseline: 1991-2020 Water Year Average (WYA)
# ============================================================

# Calculate the common 1991-2020 Water Year Average baseline
WYA_baseline = climate_summary.loc[
    BASELINE_START:BASELINE_END,
    "Water_Year_Temperature"
].mean()

print(
    f"\nCommon baseline (1991-2020 Water Year Average): "
    f"{WYA_baseline:.3f} °C"
)

# Apply the same WYA baseline to all three temperature variables
temp_columns = [
    "May_Oct_Temperature",
    "Nov_Oct_Temperature",
    "Water_Year_Temperature",
]

for col in temp_columns:

    climate_summary[col + "_Anomaly"] = (
        climate_summary[col] - WYA_baseline
    )

    print(f"{col} anomaly calculated using common WYA baseline.")

climate_summary.to_csv(
    OUTPUT_DIR / "BCSD5_climate_summary.csv"
)

# ============================================================
# 5. READ AAR DATA FOR FIVE GLACIERS
# ============================================================

aar_raw = pd.read_csv(AAR_FILE)

aar_raw.columns = aar_raw.columns.str.strip()

# Rename columns for plotting
glacier_columns = {
    "Hayden_AAR_interp": "Hayden",
    "Collier_AAR_interp": "Collier",
    "Eliot_AAR_interp": "Eliot",
    "Reid_AAR_interp": "Reid",
    "Jefferson_AAR_interp": "Jefferson",
}

# Keep year and five glacier AAR columns
aar_raw = aar_raw[["YEAR"] + list(glacier_columns.keys())].copy()

aar_raw = aar_raw.rename(columns={
    "YEAR": "Year",
    **glacier_columns
})

aar_raw["Year"] = pd.to_numeric(
    aar_raw["Year"], errors="coerce"
)

aar_raw = aar_raw.dropna(subset=["Year"])
aar_raw["Year"] = aar_raw["Year"].astype(int)

for glacier in glacier_columns.values():
    aar_raw[glacier] = pd.to_numeric(
        aar_raw[glacier], errors="coerce"
    )

# Convert wide format to long format
aar = aar_raw.melt(
    id_vars="Year",
    var_name="Glacier",
    value_name="AAR"
)

# Remove missing AAR values
aar = aar.dropna(subset=["AAR"])

print("\n===== AAR DATA =====")
print(
    aar.groupby("Glacier").agg(
        N=("AAR", "count"),
        First_Year=("Year", "min"),
        Last_Year=("Year", "max"),
    )
)

# Match AAR with climate data
merged = aar.merge(
    climate_summary,
    on="Year",
    how="left"
).sort_values(["Glacier", "Year"])

merged.to_csv(
    OUTPUT_DIR / "BCSD5_AAR_climate_matched.csv",
    index=False
)

# ============================================================
# 6. POOLED REGRESSION + PLOT
# ============================================================

GLACIER_COLORS = {
    "Collier": "tab:blue",
    "Eliot": "tab:orange",
    "Hayden": "tab:green",
    "Jefferson": "tab:red",
    "Reid": "tab:purple",
}

def perform_pooled_regression(
    data,
    x_column,
    x_label,
    output_name,
    plot_title,
):
    valid = data[["Glacier", "Year", "AAR", x_column]].dropna()

    if len(valid) < 3:
        print(f"Skipping {x_column}: fewer than 3 observations.")
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
    print(f"BCSD5 Regression: AAR vs {x_column}")
    print("===================================")
    print(f"N: {len(valid)}")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R²: {r_squared:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Slope standard error: {stderr:.6f}")

    fig, ax = plt.subplots(figsize=(8, 6))

    for glacier, group in valid.groupby("Glacier"):
        ax.scatter(
            group[x_column],
            group["AAR"],
            s=35,
            alpha=0.65,
            edgecolors="none",
            color=GLACIER_COLORS.get(glacier, "gray"),
            label=glacier,
        )

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    ax.plot(
        x_fit,
        y_fit,
        color="black",
        linewidth=2.5,
        label="Pooled linear regression",
    )

    stats_label = (
        f"N = {len(valid)}\n"
        f"P-value = {p_value:.4g}\n"
        f"R² = {r_squared:.3f}"
    )

    ax.text(
        0.03, 0.97,
        stats_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("AAR", fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{output_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return {
        "Climate_Variable": x_column,
        "N": len(valid),
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_squared,
        "p_value": p_value,
        "Slope_SE": stderr,
    }


# ============================================================
# 7. RUN REGRESSIONS
# ============================================================

regression_results = []

predictors = [
    (
        "May_Oct_Temperature_Anomaly",
        "May-October temperature anomaly (°C; BCSD5 1991-2020 WYA baseline)",
        "AAR_vs_BCSD5_May_Oct_Temperature",
        "AAR vs BCSD5 May-October Temperature",
    ),
    (
        "Nov_Oct_Temperature_Anomaly",
        "November-October temperature anomaly (°C; BCSD5 1991-2020 WYA baseline)",
        "AAR_vs_BCSD5_Nov_Oct_Temperature",
        "AAR vs BCSD5 November-October Temperature",
    ),
    (
        "Water_Year_Temperature_Anomaly",
        "Water-year temperature anomaly (°C; BCSD5 1991-2020 WYA baseline)",
        "AAR_vs_BCSD5_Water_Year_Temperature",
        "AAR vs BCSD5 Water-Year Temperature",
    ),
]

for predictor, x_label, output_name, title in predictors:
    result = perform_pooled_regression(
        merged,
        x_column=predictor,
        x_label=x_label,
        output_name=output_name,
        plot_title=title,
    )
    if result is not None:
        regression_results.append(result)

regression_results = pd.DataFrame(regression_results)
regression_results.to_csv(
    OUTPUT_DIR / "BCSD5_regression_results.csv",
    index=False,
)

print("\n===== FINAL BCSD5 REGRESSION RESULTS =====")
print(regression_results)
print("\nAnalysis complete!")
print(f"Outputs saved to: {OUTPUT_DIR}")
