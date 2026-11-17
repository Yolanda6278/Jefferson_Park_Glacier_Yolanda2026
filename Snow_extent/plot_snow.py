# ============================================================
# Mt. Jefferson Glacier: accumulation area, glacier extent,
# and water-year mean temperature
#
# Put this script and the three CSV files in the same folder.
#
# Required files:
#   PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv
#   glacier_area.csv
#   accumulation_statistics_ver1.csv
#
# Area fields used:
#   glacier_area.csv                 -> Shape_Area (m²)
#   accumulation_statistics_ver1.csv -> SUM_Shape_Area (m²)
#
# Water year is labelled by its ending year:
#   WY 2023 = October 2022 to September 2023.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PRISM_FILE = BASE_DIR / (
    "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv"
)
GLACIER_FILE = BASE_DIR / "glacier_area.csv"
ACCUMULATION_FILE = BASE_DIR / "accumulation_statistics_ver1.csv"

OUTPUT_DIR = BASE_DIR / "snow_glacier_temperature_output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FIGURE = OUTPUT_DIR / "Accumulation_Glacier_Extent_Temperature.png"


# ------------------------------------------------------------
# 2. READ PRISM AND CALCULATE WATER-YEAR MEAN TEMPERATURE
# ------------------------------------------------------------

# Skip the 11 metadata rows at the start of the PRISM CSV.
climate = pd.read_csv(PRISM_FILE, skiprows=10)
climate.columns = climate.columns.str.strip()

# Expected PRISM header:
# Date,ppt (mm),tmin (degrees C),tmean (degrees C),tmax (degrees C)
if "Date" not in climate.columns or "tmean (degrees C)" not in climate.columns:
    raise ValueError(
        "Unexpected PRISM column names. Found: "
        + ", ".join(climate.columns.astype(str))
    )

climate["Date"] = pd.to_datetime(
    climate["Date"].astype(str).str.strip(),
    format="%Y-%m",
    errors="coerce"
)
climate["Tmean"] = pd.to_numeric(
    climate["tmean (degrees C)"],
    errors="coerce"
)

climate = climate.dropna(subset=["Date", "Tmean"]).copy()
climate["Year"] = climate["Date"].dt.year
climate["Month"] = climate["Date"].dt.month

# Oct-Dec are assigned to the following water year.
climate["Water_Year"] = climate["Year"]
climate.loc[climate["Month"] >= 10, "Water_Year"] += 1

wy_temperature = (
    climate.groupby("Water_Year")
    .agg(
        Water_Year_Mean_Temperature=("Tmean", "mean"),
        Month_Count=("Tmean", "count")
    )
)

# Keep only complete water years (12 monthly values).
wy_temperature.loc[
    wy_temperature["Month_Count"] != 12,
    "Water_Year_Mean_Temperature"
] = np.nan

wy_temperature.index.name = "Year"
wy_temperature = wy_temperature.reset_index()

print("\nWater-year temperature:")
print(wy_temperature.head())


# ------------------------------------------------------------
# 3. READ ANNUAL ACCUMULATION AREA
# ------------------------------------------------------------

acc = pd.read_csv(ACCUMULATION_FILE)
acc.columns = acc.columns.str.strip()

if not {"YEAR", "SUM_Shape_Area"}.issubset(acc.columns):
    raise ValueError(
        "accumulation_statistics_ver1.csv needs YEAR and SUM_Shape_Area. "
        f"Found: {acc.columns.tolist()}"
    )

acc["Year"] = pd.to_numeric(acc["YEAR"], errors="coerce")
acc["Accumulation_Area_m2"] = pd.to_numeric(
    acc["SUM_Shape_Area"], errors="coerce"
)
acc = acc.dropna(subset=["Year", "Accumulation_Area_m2"]).copy()
acc["Year"] = acc["Year"].astype(int)

# Sum any duplicate records belonging to the same year.
acc = (
    acc.groupby("Year", as_index=False)["Accumulation_Area_m2"]
    .sum()
    .sort_values("Year")
)

print("\nAccumulation-area data:")
print(acc.head())
print("Number of years:", len(acc))


# ------------------------------------------------------------
# 4. READ GLACIER EXTENT AREA
# ------------------------------------------------------------

glacier = pd.read_csv(GLACIER_FILE)
glacier.columns = glacier.columns.str.strip()

if not {"Year", "Shape_Area"}.issubset(glacier.columns):
    raise ValueError(
        "glacier_area.csv needs Year and Shape_Area. "
        f"Found: {glacier.columns.tolist()}"
    )

glacier["Year"] = pd.to_numeric(glacier["Year"], errors="coerce")
glacier["Glacier_Area_m2"] = pd.to_numeric(
    glacier["Shape_Area"], errors="coerce"
)
glacier = glacier.dropna(subset=["Year", "Glacier_Area_m2"]).copy()
glacier["Year"] = glacier["Year"].astype(int)

# Sum duplicate outline records if any.
glacier = (
    glacier.groupby("Year", as_index=False)["Glacier_Area_m2"]
    .sum()
    .sort_values("Year")
)

print("\nGlacier extent data:")
print(glacier)


# ------------------------------------------------------------
# 5. MATCH ACCUMULATION AREA WITH WATER-YEAR TEMPERATURE
# ------------------------------------------------------------

acc_temp = acc.merge(
    wy_temperature[["Year", "Water_Year_Mean_Temperature"]],
    on="Year",
    how="left"
).sort_values("Year")

missing_temp_years = acc_temp.loc[
    acc_temp["Water_Year_Mean_Temperature"].isna(), "Year"
].tolist()

if missing_temp_years:
    print(
        "\nWARNING: Missing complete water-year temperature for:",
        missing_temp_years
    )

acc_temp.to_csv(
    OUTPUT_DIR / "Accumulation_Area_and_Water_Year_Temperature.csv",
    index=False
)


# ------------------------------------------------------------
# 6. PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 8))

# Accumulation-area line: connects observed annual values.
# Years absent from the CSV (e.g. 2012) remain gaps in the data;
# no missing values are interpolated.
ax.plot(
    acc["Year"],
    acc["Accumulation_Area_m2"],
    color="0.35",
    linewidth=1.3,
    alpha=0.85,
    zorder=2,
    label="Accumulation area"
)

# Linear trend for annual accumulation/snow extent.
# Fit only to years with measured values; missing years are not filled.
if len(acc) >= 2 and acc["Year"].nunique() >= 2:
    snow_trend = linregress(
        acc["Year"].astype(float),
        acc["Accumulation_Area_m2"].astype(float)
    )
    trend_years = np.linspace(acc["Year"].min(), acc["Year"].max(), 200)
    trend_areas = snow_trend.intercept + snow_trend.slope * trend_years

    ax.plot(
        trend_years,
        trend_areas,
        color="#4A4A4A",       # muted orange / amber
        linestyle="--",
        linewidth=2.5,
        alpha=0.95,
        zorder=3,
        label="Linear trend (snow extent)"
    )

    print("\\n===== SNOW EXTENT LINEAR TREND =====")
    print(f"Slope: {snow_trend.slope:.2f} m²/year")
    print(f"R²: {snow_trend.rvalue ** 2:.4f}")
    print(f"p-value: {snow_trend.pvalue:.6f}")
else:
    print("Not enough annual snow-area observations to calculate a linear trend.")

# Accumulation-area points coloured by water-year mean temperature.
points = acc_temp.dropna(
    subset=["Water_Year_Mean_Temperature"]
)

sc = ax.scatter(
    points["Year"],
    points["Accumulation_Area_m2"],
    c=points["Water_Year_Mean_Temperature"],
    cmap="coolwarm",
    s=85,
    edgecolors="white",
    linewidths=0.6,
    alpha=0.95,
    zorder=4
)

# Glacier extent: black line and black markers.
ax.plot(
    glacier["Year"],
    glacier["Glacier_Area_m2"],
    color="black",
    linewidth=1.6,
    marker="o",
    markersize=7,
    markerfacecolor="black",
    markeredgecolor="black",
    zorder=3,
    label="Glacier extent"
)

ax.set_title(
    "Variation in Accumulation Area and Glacier Extent Through Time",
    fontsize=16,
    fontweight="bold",
    pad=14
)
ax.set_xlabel("Year", fontsize=13, fontweight="bold")
ax.set_ylabel("Area (m²)", fontsize=13, fontweight="bold")

ax.grid(True, linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

# Set x-axis limits and 5-year ticks based on both datasets.
years_all = pd.concat(
    [acc["Year"], glacier["Year"]],
    ignore_index=True
).dropna()

if not years_all.empty:
    first_year = int(years_all.min())
    last_year = int(years_all.max())
    ax.set_xlim(first_year - 0.7, last_year + 0.7)
    first_tick = (first_year // 5) * 5
    ax.set_xticks(np.arange(first_tick, last_year + 1, 5))

# Temperature colour bar.
cbar = fig.colorbar(sc, ax=ax, pad=0.035)
cbar.set_label(
    "Water-year average temperature (°C)",
    fontsize=12
)

# Legend for the two area series.
ax.legend(loc="upper right", frameon=True, framealpha=0.95)

fig.tight_layout()
fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")

print("\nFigure saved to:", OUTPUT_FIGURE)

plt.show()


# ------------------------------------------------------------
# 7. EXPORT A COMBINED TABLE
# ------------------------------------------------------------

combined = (
    acc.merge(glacier, on="Year", how="outer")
    .merge(
        wy_temperature[["Year", "Water_Year_Mean_Temperature"]],
        on="Year",
        how="outer"
    )
    .sort_values("Year")
)

combined.to_csv(
    OUTPUT_DIR / "Combined_Snow_Glacier_Temperature_Data.csv",
    index=False
)

print("\nCombined data:")
print(combined.to_string(index=False))
print("\nFinished. Outputs are in:", OUTPUT_DIR)
