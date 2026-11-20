# ============================================================
# Mt. Jefferson Glacier
# Climate–AAR Linear Regression
#
# Temperature predictors:
#   1. Calendar-year mean temperature anomaly (Jan-Dec)
#   2. Water-year mean temperature anomaly (Oct-Sep)
#
# Water year is labelled by its ending year:
#   Water Year 2020 = Oct 2019-Sep 2020
#
# Summer temperature and winter precipitation are omitted for now.
# Annual precipitation regression is retained.
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

AAR_FILE = BASE_DIR / "AAR_interpolation_extrapolation_19Nov.csv"
PRISM_FILE = BASE_DIR / (
    "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Temperature anomaly baseline period
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
# 3. CALENDAR-YEAR CLIMATE (JANUARY-DECEMBER)
# ============================================================

annual_climate = climate.groupby("Year").agg(
    Annual_Temperature=("Temperature", "mean"),
    Annual_Precipitation=("Precipitation", "sum"),
    Annual_Temperature_Months=("Temperature", "count")
)

# Require all 12 monthly temperature values for a complete annual mean
annual_climate.loc[
    annual_climate["Annual_Temperature_Months"] != 12,
    "Annual_Temperature"
] = np.nan


# ============================================================
# 4. WATER-YEAR CLIMATE (OCTOBER-SEPTEMBER)
# ============================================================

wy = climate.copy()

# October-December belong to the following water year
wy["Water_Year"] = wy["Year"]
wy.loc[wy["Month"] >= 10, "Water_Year"] += 1

water_year_climate = wy.groupby("Water_Year").agg(
    Water_Year_Temperature=("Temperature", "mean"),
    Water_Year_Precipitation=("Precipitation", "sum"),
    Water_Year_Temperature_Months=("Temperature", "count")
)

# Require all 12 monthly temperature values for a complete water-year mean
water_year_climate.loc[
    water_year_climate["Water_Year_Temperature_Months"] != 12,
    "Water_Year_Temperature"
] = np.nan

water_year_climate.index.name = "Year"

# ============================================================
# 4.5 NOVEMBER-OCTOBER CLIMATE (NOVEMBER-OCTOBER)
# ============================================================

nov_oct = climate.copy()

# November-December belong to the following year
nov_oct["Nov_Oct_Year"] = nov_oct["Year"]

nov_oct.loc[
    nov_oct["Month"] >= 11,
    "Nov_Oct_Year"
] += 1

nov_oct_climate = nov_oct.groupby("Nov_Oct_Year").agg(
    Nov_Oct_Temperature=("Temperature", "mean"),
    Nov_Oct_Precipitation=("Precipitation", "sum"),
    Nov_Oct_Temperature_Months=("Temperature", "count")
)

# Require all 12 monthly temperature values
nov_oct_climate.loc[
    nov_oct_climate["Nov_Oct_Temperature_Months"] != 12,
    "Nov_Oct_Temperature"
] = np.nan

nov_oct_climate.index.name = "Year"

# ============================================================
# 4.6 WINTER PRECIPITATION (NOVEMBER-APRIL / NOVEMBER-MAY)
# ============================================================

winter_precip = climate.copy()

# November-December belong to the following year
winter_precip["Climate_Year"] = winter_precip["Year"]

winter_precip.loc[
    winter_precip["Month"] >= 11,
    "Climate_Year"
] += 1

# November-April precipitation
nov_apr = winter_precip[
    winter_precip["Month"].isin([11, 12, 1, 2, 3, 4])
].groupby("Climate_Year").agg(
    Nov_Apr_Precipitation=("Precipitation", "sum"),
    Nov_Apr_Precipitation_Months=("Precipitation", "count")
)

# Require all 6 months
nov_apr.loc[
    nov_apr["Nov_Apr_Precipitation_Months"] != 6,
    "Nov_Apr_Precipitation"
] = np.nan


# November-May precipitation
nov_may = winter_precip[
    winter_precip["Month"].isin([11, 12, 1, 2, 3, 4, 5])
].groupby("Climate_Year").agg(
    Nov_May_Precipitation=("Precipitation", "sum"),
    Nov_May_Precipitation_Months=("Precipitation", "count")
)

# Require all 7 months
nov_may.loc[
    nov_may["Nov_May_Precipitation_Months"] != 7,
    "Nov_May_Precipitation"
] = np.nan


# Combine both precipitation periods
winter_precip_climate = nov_apr.join(
    nov_may,
    how="outer"
)

winter_precip_climate.index.name = "Year"

# ============================================================
# 5. TEMPERATURE NORMALISATION (ANOMALIES)
# ============================================================

# Baseline means are calculated over the specified calendar-year period.
annual_baseline = annual_climate.loc[
    BASELINE_START:BASELINE_END, "Annual_Temperature"
].mean()

water_year_baseline = water_year_climate.loc[
    BASELINE_START:BASELINE_END, "Water_Year_Temperature"
].mean()

nov_oct_baseline = nov_oct_climate.loc[
    BASELINE_START:BASELINE_END, "Nov_Oct_Temperature"
].mean()

annual_climate["Annual_Temperature_Anomaly"] = (
    annual_climate["Annual_Temperature"] - annual_baseline
)

water_year_climate["Water_Year_Temperature_Anomaly"] = (
    water_year_climate["Water_Year_Temperature"] - water_year_baseline
)

nov_oct_climate["Nov_Oct_Temperature_Anomaly"] = (
    nov_oct_climate["Nov_Oct_Temperature"] - nov_oct_baseline
)

print("\\n===== TEMPERATURE BASELINES =====")
print(f"Calendar-year mean ({BASELINE_START}-{BASELINE_END}): {annual_baseline:.3f} °C")
print(f"Water-year mean ({BASELINE_START}-{BASELINE_END}): {water_year_baseline:.3f} °C")


# ============================================================
# 6. COMBINE AND EXPORT CLIMATE DATA
# ============================================================

climate_summary = (
    annual_climate
    .join(water_year_climate, how="outer")
    .join(nov_oct_climate, how="outer")
    .join(winter_precip_climate, how="outer")
    .sort_index()
)

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
# 9. CLIMATE TIME-SERIES PLOTS
# ============================================================

# Calendar-year temperature anomaly
plt.figure(figsize=(12, 5))
plt.plot(
    climate_summary.index,
    climate_summary["Annual_Temperature_Anomaly"],
    linewidth=1.5, marker="."
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Year")
plt.ylabel("Temperature anomaly (°C)")
plt.title(f"Calendar-year Temperature Anomaly ({BASELINE_START}-{BASELINE_END} baseline)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Annual_Temperature_Anomaly.png", dpi=300)
plt.show()

# Water-year temperature anomaly
plt.figure(figsize=(12, 5))
plt.plot(
    climate_summary.index,
    climate_summary["Water_Year_Temperature_Anomaly"],
    linewidth=1.5, marker="."
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Water year (Oct-Sep)")
plt.ylabel("Temperature anomaly (°C)")
plt.title(f"Water-year Temperature Anomaly ({BASELINE_START}-{BASELINE_END} baseline)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Water_Year_Temperature_Anomaly.png", dpi=300)
plt.show()


# ============================================================
# 10. AAR TIME SERIES
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(aar["Year"], aar["AAR"], marker="o", linewidth=1.5)
plt.xlabel("Year")
plt.ylabel("AAR")
plt.title("Accumulation Area Ratio - Jefferson Park Glacier")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "AAR_timeseries.png", dpi=300)
plt.show()


# ============================================================
# 11. REGRESSION FUNCTION
# ============================================================

def perform_regression(
    data,
    x_column,
    y_column="AAR",
    x_label=None,
    y_label="AAR",
    output_name=None,
    point_color="blue",
    plot_title=None
):
    """
    Run a simple linear regression and plot AAR against a climate variable.
    P-value and R-squared are displayed in the upper-right legend.
    """

    valid = data[[x_column, y_column]].dropna()

    if len(valid) < 3:
        print(f"Skipping {x_column}: fewer than 3 valid observations.")
        return None

    x = valid[x_column]
    y = valid[y_column]
    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    p_value = result.pvalue
    stderr = result.stderr

    print("\\n===================================")
    print(f"Regression: {y_column} vs {x_column}")
    print("===================================")
    print(f"N: {len(valid)}")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R²: {r_squared:.4f}")
    print(f"p-value: {p_value:.6f}")
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
        x_label if x_label else x_column,
        fontsize=12,
        fontweight="bold"
    )
    ax.set_ylabel(
        y_label,
        fontsize=12,
        fontweight="bold"
    )

    ax.set_title(
        plot_title if plot_title else f"AAR vs {x_column}",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    # Add statistics as a legend entry (upper-right corner)
    stats_label = f"P-value = {p_value:.4f}\nR² = {r_squared:.2f}"
    ax.plot(
        [], [],
        linestyle="none",
        marker="",
        label=stats_label
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_name:
        plt.savefig(
            OUTPUT_DIR / f"{output_name}.png",
            dpi=300,
            bbox_inches="tight"
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
# 12. REGRESSIONS
# ============================================================

regression_results = []

# 1) AAR vs calendar-year temperature anomaly (Jan-Dec)
result = perform_regression(
    merged,
    x_column="Annual_Temperature_Anomaly",
    x_label="Temperature anomaly (°C)",
    output_name="AAR_vs_Annual_Temperature_Anomaly",
    point_color="crimson",
    plot_title="AAR vs Annual Temperature Average"
)
if result is not None:
    regression_results.append(result)

# 2) AAR vs water-year temperature anomaly (Oct-Sep)
result = perform_regression(
    merged,
    x_column="Water_Year_Temperature_Anomaly",
    x_label="Temperature anomaly (°C)",
    output_name="AAR_vs_Water_Year_Temperature_Anomaly",
    point_color="blue",
    plot_title="AAR vs Water Year Average"
)
if result is not None:
    regression_results.append(result)

# 3) AAR vs November-October temperature anomaly
result = perform_regression(
    merged,
    x_column="Nov_Oct_Temperature_Anomaly",
    x_label="Temperature anomaly (°C)",
    output_name="AAR_vs_Nov_Oct_Temperature_Anomaly",
    point_color="darkorange",
    plot_title="AAR vs November-October Temperature Average"
)

if result is not None:
    regression_results.append(result)

# 4) AAR vs annual precipitation (retained)
result = perform_regression(
    merged,
    x_column="Annual_Precipitation",
    x_label="Calendar-year precipitation (mm)",
    output_name="AAR_vs_Annual_Precipitation",
    point_color="purple"
)
if result is not None:
    regression_results.append(result)

# Summer temperature and winter precipitation are intentionally not run.

# 5) AAR vs November-April precipitation
result = perform_regression(
    merged,
    x_column="Nov_Apr_Precipitation",
    x_label="November-April precipitation (mm)",
    output_name="AAR_vs_Nov_Apr_Precipitation",
    point_color="royalblue",
    plot_title="AAR vs November-April Precipitation"
)

if result is not None:
    regression_results.append(result)


# 6) AAR vs November-May precipitation
result = perform_regression(
    merged,
    x_column="Nov_May_Precipitation",
    x_label="November-May precipitation (mm)",
    output_name="AAR_vs_Nov_May_Precipitation",
    point_color="teal",
    plot_title="AAR vs November-May Precipitation"
)

if result is not None:
    regression_results.append(result)

# ============================================================
# 13. EXPORT REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame(regression_results)
regression_results.to_csv(OUTPUT_DIR / "Regression_results_normalised.csv", index=False)

print("\\n===== FINAL REGRESSION RESULTS =====")
print(regression_results)
print("\\nAnalysis complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")
