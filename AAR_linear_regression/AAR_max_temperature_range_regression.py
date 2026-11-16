# ============================================================
# Mt. Jefferson Glacier
# AAR vs Annual Maximum Temperature Range Regression
#
# Two temperature metrics:
#   1. Calendar year: Jan-Dec
#   2. Water year: Oct-Sep (Oct-Dec assigned to the following year)
#
# Maximum temperature range is defined as:
#   max(Tmax) - min(Tmin)
# across the selected 12 months in each year.
#
# Temperature anomaly baseline:
#   1991-2020 mean of the annual maximum temperature range.
#
# Outputs are saved separately in:
#   output_max_temperature_range/
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

OUTPUT_DIR = BASE_DIR / "output_max_temperature_range"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_START = 1991
BASELINE_END = 2020

ANALYSIS_START = 1985
ANALYSIS_END = 2026


# ============================================================
# 2. READ PRISM DATA
# ============================================================

# The PRISM file has 11 metadata lines before the data header
climate = pd.read_csv(PRISM_FILE, skiprows=11)

climate.columns = [
    "Date",
    "Precipitation",
    "Tmin",
    "Tmean",
    "Tmax"
]

climate["Date"] = pd.to_datetime(
    climate["Date"],
    format="%Y-%m",
    errors="coerce"
)

for col in ["Tmin", "Tmean", "Tmax"]:
    climate[col] = pd.to_numeric(climate[col], errors="coerce")

climate["Year"] = climate["Date"].dt.year
climate["Month"] = climate["Date"].dt.month

print("\n===== PRISM DATA =====")
print(climate.head())
print("Climate period:", climate["Date"].min(), "to", climate["Date"].max())


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

aar = aar[
    (aar["Year"] >= ANALYSIS_START) &
    (aar["Year"] <= ANALYSIS_END)
]

print("\n===== AAR DATA =====")
print(aar.head())
print("Missing AAR years:",
      sorted(set(range(ANALYSIS_START, ANALYSIS_END + 1)) - set(aar["Year"])))


# ============================================================
# 4. CALENDAR-YEAR MAXIMUM TEMPERATURE RANGE
# ============================================================

# For each calendar year:
# annual range = highest monthly Tmax - lowest monthly Tmin
calendar = climate[
    (climate["Year"] >= ANALYSIS_START) &
    (climate["Year"] <= ANALYSIS_END)
].copy()

calendar_stats = calendar.groupby("Year").agg(
    Tmax_max=("Tmax", "max"),
    Tmin_min=("Tmin", "min"),
    Month_count=("Date", "count")
)

# Require all 12 months
calendar_stats.loc[
    calendar_stats["Month_count"] != 12,
    ["Tmax_max", "Tmin_min"]
] = np.nan

calendar_stats["Max_Temperature_Range"] = (
    calendar_stats["Tmax_max"] - calendar_stats["Tmin_min"]
)

calendar_stats = calendar_stats.rename_axis("Year").reset_index()


# ============================================================
# 5. WATER-YEAR MAXIMUM TEMPERATURE RANGE
# ============================================================

# Water year convention:
# October-December belong to the following water year.
water = climate.copy()
water["Water_Year"] = np.where(
    water["Month"] >= 10,
    water["Year"] + 1,
    water["Year"]
)

water = water[
    (water["Water_Year"] >= ANALYSIS_START) &
    (water["Water_Year"] <= ANALYSIS_END)
].copy()

water_stats = water.groupby("Water_Year").agg(
    Tmax_max=("Tmax", "max"),
    Tmin_min=("Tmin", "min"),
    Month_count=("Date", "count")
)

# Require all 12 months
water_stats.loc[
    water_stats["Month_count"] != 12,
    ["Tmax_max", "Tmin_min"]
] = np.nan

water_stats["Max_Temperature_Range"] = (
    water_stats["Tmax_max"] - water_stats["Tmin_min"]
)

water_stats = water_stats.rename_axis("Year").reset_index()


# ============================================================
# 6. CALCULATE ANOMALIES (1991-2020 BASELINE)
# ============================================================

def add_anomaly(df, value_column):
    baseline_mean = df.loc[
        (df["Year"] >= BASELINE_START) &
        (df["Year"] <= BASELINE_END),
        value_column
    ].mean()

    df["Temperature_Range_Anomaly"] = (
        df[value_column] - baseline_mean
    )

    print(
        f"{value_column} baseline ({BASELINE_START}-{BASELINE_END}): "
        f"{baseline_mean:.2f} °C"
    )

    return df


calendar_stats = add_anomaly(
    calendar_stats,
    "Max_Temperature_Range"
)

water_stats = add_anomaly(
    water_stats,
    "Max_Temperature_Range"
)


# ============================================================
# 7. MATCH AAR WITH CLIMATE METRICS
# ============================================================

calendar_matched = aar.merge(
    calendar_stats,
    on="Year",
    how="left"
)

water_matched = aar.merge(
    water_stats,
    on="Year",
    how="left"
)

calendar_matched.to_csv(
    OUTPUT_DIR / "AAR_Calendar_Year_Max_Temperature_Range.csv",
    index=False
)

water_matched.to_csv(
    OUTPUT_DIR / "AAR_Water_Year_Max_Temperature_Range.csv",
    index=False
)


# ============================================================
# 8. REGRESSION AND PLOT FUNCTION
# ============================================================

def perform_regression(
    data,
    x_column,
    plot_title,
    output_name,
    point_color
):

    valid = data[["AAR", x_column]].dropna()

    if len(valid) < 3:
        print(f"Skipping {plot_title}: fewer than 3 valid observations.")
        return None

    x = valid[x_column]
    y = valid["AAR"]

    reg = linregress(x, y)

    slope = reg.slope
    intercept = reg.intercept
    r_squared = reg.rvalue ** 2
    p_value = reg.pvalue
    stderr = reg.stderr

    print("\n===================================")
    print(plot_title)
    print("===================================")
    print(f"N = {len(valid)}")
    print(f"Slope = {slope:.6f}")
    print(f"Intercept = {intercept:.6f}")
    print(f"R² = {r_squared:.4f}")
    print(f"P-value = {p_value:.6f}")
    print(f"Slope SE = {stderr:.6f}")

    # Plot
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
        "Maximum temperature range anomaly (°C)",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_ylabel("AAR", fontsize=12, fontweight="bold")

    ax.set_title(
        plot_title,
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    stats_label = (
        f"P-value = {p_value:.4f}\n"
        f"\nR² = {r_squared:.2f}"
    )

    ax.plot([], [], linestyle="none", label=stats_label)

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
        "Metric": plot_title,
        "N": len(valid),
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_squared,
        "p_value": p_value,
        "Slope_SE": stderr
    }


# ============================================================
# 9. RUN TWO REGRESSIONS
# ============================================================

results = []

# Calendar year: January-December
result_calendar = perform_regression(
    data=calendar_matched,
    x_column="Temperature_Range_Anomaly",
    plot_title="AAR vs Calendar Year Maximum Temperature Range",
    output_name="AAR_vs_Calendar_Year_Max_Temperature_Range",
    point_color="crimson"
)

if result_calendar is not None:
    result_calendar["Year_Type"] = "Calendar year"
    results.append(result_calendar)

# Water year: October-September
result_water = perform_regression(
    data=water_matched,
    x_column="Temperature_Range_Anomaly",
    plot_title="AAR vs Water Year Maximum Temperature Range",
    output_name="AAR_vs_Water_Year_Max_Temperature_Range",
    point_color="blue"
)

if result_water is not None:
    result_water["Year_Type"] = "Water year"
    results.append(result_water)


# ============================================================
# 10. EXPORT RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "Regression_results_max_temperature_range.csv",
    index=False
)

print("\n===== REGRESSION SUMMARY =====")
print(results_df)

print("\nAnalysis complete!")
print(f"Outputs saved to: {OUTPUT_DIR}")
