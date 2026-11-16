# ============================================================
# Mt. Jefferson Glacier
# Lagged Climate-AAR Linear Regression
#
# Tests climate predictors at lag 0-3:
#   1. Calendar-year temperature anomaly (Jan-Dec)
#   2. Water-year temperature anomaly (Oct-Sep)
#   3. Calendar-year precipitation (Jan-Dec)
#
# Lag definition:
#   Lag 0: climate in AAR observation year y
#   Lag 1: climate in year y-1
#   Lag 2: climate in year y-2
#   Lag 3: climate in year y-3
#
# Water year is labelled by its ending year:
#   Water Year 2020 = Oct 2019-Sep 2020
#
# Notes:
# - AAR(y) is regressed against climate(y-lag).
# - Each model uses its available paired observations (listwise deletion
#   for that model); N is reported so differing sample sizes are visible.
# - These are exploratory ordinary least-squares regressions. Temporal
#   autocorrelation and interpolated/extrapolated AAR may affect inference.
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

OUTPUT_DIR = BASE_DIR / "output_lagged_regression"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_START = 1991
BASELINE_END = 2020
MAX_LAG = 3


# ============================================================
# 2. READ PRISM DATA
# ============================================================

climate = pd.read_csv(PRISM_FILE, skiprows=11)

climate.columns = [
    "Date",
    "Precipitation",
    "Tmin",
    "Tmean",
    "Tmax"
]

climate["Date"] = pd.to_datetime(climate["Date"], format="%Y-%m")

for col in ["Precipitation", "Tmin", "Tmean", "Tmax"]:
    climate[col] = pd.to_numeric(climate[col], errors="coerce")

climate["Year"] = climate["Date"].dt.year
climate["Month"] = climate["Date"].dt.month
climate["Temperature"] = climate["Tmean"]

print("\\n===== PRISM CLIMATE DATA =====")
print(climate.head())
print("Climate period:", climate["Date"].min(), "to", climate["Date"].max())


# ============================================================
# 3. CALENDAR-YEAR CLIMATE (JANUARY-DECEMBER)
# ============================================================

annual_climate = climate.groupby("Year").agg(
    Annual_Temperature=("Temperature", "mean"),
    Annual_Precipitation=("Precipitation", "sum"),
    Annual_Temperature_Months=("Temperature", "count"),
    Annual_Precipitation_Months=("Precipitation", "count")
)

annual_climate.loc[
    annual_climate["Annual_Temperature_Months"] != 12,
    "Annual_Temperature"
] = np.nan

annual_climate.loc[
    annual_climate["Annual_Precipitation_Months"] != 12,
    "Annual_Precipitation"
] = np.nan


# ============================================================
# 4. WATER-YEAR CLIMATE (OCTOBER-SEPTEMBER)
# ============================================================

wy = climate.copy()
wy["Water_Year"] = wy["Year"]
wy.loc[wy["Month"] >= 10, "Water_Year"] += 1

water_year_climate = wy.groupby("Water_Year").agg(
    Water_Year_Temperature=("Temperature", "mean"),
    Water_Year_Precipitation=("Precipitation", "sum"),
    Water_Year_Temperature_Months=("Temperature", "count"),
    Water_Year_Precipitation_Months=("Precipitation", "count")
)

water_year_climate.loc[
    water_year_climate["Water_Year_Temperature_Months"] != 12,
    "Water_Year_Temperature"
] = np.nan

water_year_climate.loc[
    water_year_climate["Water_Year_Precipitation_Months"] != 12,
    "Water_Year_Precipitation"
] = np.nan

water_year_climate.index.name = "Year"


# ============================================================
# 5. TEMPERATURE ANOMALIES
# ============================================================

annual_baseline = annual_climate.loc[
    BASELINE_START:BASELINE_END, "Annual_Temperature"
].mean()

water_year_baseline = water_year_climate.loc[
    BASELINE_START:BASELINE_END, "Water_Year_Temperature"
].mean()

annual_climate["Annual_Temperature_Anomaly"] = (
    annual_climate["Annual_Temperature"] - annual_baseline
)

water_year_climate["Water_Year_Temperature_Anomaly"] = (
    water_year_climate["Water_Year_Temperature"] - water_year_baseline
)

print("\\n===== TEMPERATURE BASELINES =====")
print(f"Calendar-year mean ({BASELINE_START}-{BASELINE_END}): {annual_baseline:.3f} °C")
print(f"Water-year mean ({BASELINE_START}-{BASELINE_END}): {water_year_baseline:.3f} °C")


# ============================================================
# 6. COMBINE AND EXPORT CLIMATE SUMMARY
# ============================================================

climate_summary = annual_climate.join(water_year_climate, how="outer").sort_index()
climate_summary.to_csv(OUTPUT_DIR / "Climate_summary_normalised.csv")

print("\\n===== CLIMATE SUMMARY =====")
print(climate_summary.head())


# ============================================================
# 7. READ AAR DATA
# ============================================================

aar = pd.read_csv(AAR_FILE)
aar.columns = aar.columns.str.strip()

aar["YEAR"] = pd.to_numeric(aar["YEAR"], errors="coerce")
aar["AAR"] = pd.to_numeric(aar["AAR"], errors="coerce")
aar = aar.dropna(subset=["YEAR", "AAR"])
aar["Year"] = aar["YEAR"].astype(int)
aar = aar[["Year", "AAR"]].sort_values("Year").drop_duplicates("Year", keep="last")

print("\\n===== AAR DATA =====")
print(aar.head())
print("Missing AAR years:", sorted(set(range(1985, 2027)) - set(aar["Year"])))


# ============================================================
# 8. MATCH AAR AND CLIMATE
# ============================================================

merged = aar.merge(
    climate_summary,
    left_on="Year",
    right_index=True,
    how="left"
).sort_values("Year")

merged.to_csv(OUTPUT_DIR / "AAR_climate_matched_normalised.csv", index=False)

print("\\n===== MATCHED DATA =====")
print(merged.head())
print("Matched data shape:", merged.shape)


# ============================================================
# 9. CREATE LAGGED PREDICTORS
# ============================================================

# Shift climate series by k years, then match to AAR year y.
# For example, lag 1 means predictor value from climate year y-1.
predictor_columns = [
    "Annual_Temperature_Anomaly",
    "Water_Year_Temperature_Anomaly",
    "Annual_Precipitation"
]

for col in predictor_columns:
    for lag in range(MAX_LAG + 1):
        lag_col = f"{col}_Lag{lag}"
        merged[lag_col] = merged[col].shift(lag)

merged.to_csv(OUTPUT_DIR / "AAR_climate_with_lags.csv", index=False)


# ============================================================
# 10. REGRESSION FUNCTION
# ============================================================

def perform_regression(data, x_column, lag, x_label, output_name, point_color):
    """Run OLS regression of AAR against a climate predictor at a given lag."""

    valid = data[[x_column, "AAR"]].dropna()

    if len(valid) < 3 or valid[x_column].nunique() < 2:
        print(f"Skipping {x_column}: insufficient valid observations/variation.")
        return None

    x = valid[x_column].astype(float)
    y = valid["AAR"].astype(float)
    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    p_value = result.pvalue
    stderr = result.stderr
    n = len(valid)

    # Adjusted R-squared for simple linear regression (one predictor)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

    # AIC for Gaussian OLS, based on residual sum of squares.
    # k = 3 parameters: intercept, slope, and residual variance.
    residuals = y - (slope * x + intercept)
    rss = float(np.sum(residuals ** 2))
    rss = max(rss, np.finfo(float).tiny)
    aic = n * np.log(rss / n) + 2 * 3

    print("\\n===================================")
    print(f"Regression: AAR vs {x_column} (lag {lag})")
    print("===================================")
    print(f"N: {n}")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R²: {r_squared:.4f}")
    print(f"Adjusted R²: {adjusted_r_squared:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"AIC: {aic:.3f}")
    print(f"Standard error of slope: {stderr:.6f}")

    # Scatter and fitted line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=45, alpha=0.85, color=point_color, label="AAR")

    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="green", linewidth=2.5, label="Linear regression")

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("AAR", fontsize=12, fontweight="bold")
    ax.set_title(f"AAR vs {x_label} (Lag {lag})", fontsize=14, fontweight="bold", pad=12)

    stats_label = (
        f"N = {n}\nR² = {r_squared:.3f}\n"
        f"Adj. R² = {adjusted_r_squared:.3f}\n"
        f"p = {p_value:.4g}\nAIC = {aic:.2f}"
    )
    ax.plot([], [], linestyle="none", marker="", label=stats_label)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "Predictor": x_column.replace(f"_Lag{lag}", ""),
        "Lag": lag,
        "N": n,
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_squared,
        "Adjusted_R_squared": adjusted_r_squared,
        "p_value": p_value,
        "AIC": aic,
        "Slope_SE": stderr
    }


# ============================================================
# 11. RUN LAGGED REGRESSIONS (LAG 0-3)
# ============================================================

regression_results = []

predictor_info = {
    "Annual_Temperature_Anomaly": {
        "label": "Calendar-year temperature anomaly (°C)",
        "color": "crimson"
    },
    "Water_Year_Temperature_Anomaly": {
        "label": "Water-year temperature anomaly (°C)",
        "color": "blue"
    },
    "Annual_Precipitation": {
        "label": "Calendar-year precipitation (mm)",
        "color": "purple"
    }
}

for predictor, info in predictor_info.items():
    for lag in range(MAX_LAG + 1):
        lag_column = f"{predictor}_Lag{lag}"
        result = perform_regression(
            data=merged,
            x_column=lag_column,
            lag=lag,
            x_label=info["label"],
            output_name=f"AAR_vs_{predictor}_Lag{lag}",
            point_color=info["color"]
        )
        if result is not None:
            regression_results.append(result)


# ============================================================
# 12. EXPORT REGRESSION RESULTS
# ============================================================

results_df = pd.DataFrame(regression_results)
results_df.to_csv(OUTPUT_DIR / "Lagged_regression_results.csv", index=False)

print("\\n===== ALL LAGGED REGRESSION RESULTS =====")
print(results_df.to_string(index=False))


# ============================================================
# 13. PLOT R², ADJUSTED R², AND P-VALUE AGAINST LAG
# ============================================================

for predictor in results_df["Predictor"].unique():
    subset = results_df[results_df["Predictor"] == predictor].sort_values("Lag")

    # R-squared
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(subset["Lag"], subset["R_squared"], marker="o", linewidth=1.8)
    ax.set_xticks(range(MAX_LAG + 1))
    ax.set_xlabel("Climate lag (years)")
    ax.set_ylabel("R²")
    ax.set_title(f"R² vs Lag: {predictor}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"R_squared_vs_lag_{predictor}.png", dpi=300)
    plt.close(fig)

    # Adjusted R-squared
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(subset["Lag"], subset["Adjusted_R_squared"], marker="o", linewidth=1.8)
    ax.set_xticks(range(MAX_LAG + 1))
    ax.set_xlabel("Climate lag (years)")
    ax.set_ylabel("Adjusted R²")
    ax.set_title(f"Adjusted R² vs Lag: {predictor}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Adjusted_R_squared_vs_lag_{predictor}.png", dpi=300)
    plt.close(fig)

    # p-value (log scale helps when values vary greatly)
    fig, ax = plt.subplots(figsize=(7, 5))
    p_plot = subset["p_value"].clip(lower=1e-16)
    ax.plot(subset["Lag"], p_plot, marker="o", linewidth=1.8)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_yscale("log")
    ax.set_xticks(range(MAX_LAG + 1))
    ax.set_xlabel("Climate lag (years)")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title(f"p-value vs Lag: {predictor}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"p_value_vs_lag_{predictor}.png", dpi=300)
    plt.close(fig)


# ============================================================
# 14. SUMMARY: BEST LAG PER PREDICTOR
# ============================================================

best_lag_summary = (
    results_df.sort_values(["Predictor", "AIC"])
    .groupby("Predictor", as_index=False)
    .first()
)

best_lag_summary.to_csv(OUTPUT_DIR / "Best_lag_by_AIC.csv", index=False)

print("\\n===== BEST LAG BY AIC (LOWER IS BETTER) =====")
print(best_lag_summary.to_string(index=False))

print("\\nAnalysis complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")
