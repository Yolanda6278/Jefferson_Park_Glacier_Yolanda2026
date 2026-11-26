# ============================================================
# Mt. Jefferson Glacier: Combined Area and AAR Plot
# Two vertically stacked panels with a shared x-axis
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PRISM_FILE = BASE_DIR / (
    "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv"
)
ACCUMULATION_FILE = BASE_DIR / "accumulation_statistics_ver2_19Nov.csv"
GLACIER_FILE = BASE_DIR / "glacier_area_19Nov.csv"

OUTPUT_DIR = BASE_DIR / "combined_AAR_area_output"
OUTPUT_DIR.mkdir(exist_ok=True)

START_YEAR = 1985
END_YEAR = 2026

OUTPUT_FIGURE = OUTPUT_DIR / "Combined_AAR_and_Area.png"
OUTPUT_CSV = OUTPUT_DIR / "Combined_AAR_and_Area_Data.csv"

# ------------------------------------------------------------
# 2. PRISM WATER-YEAR MEAN TEMPERATURE
# ------------------------------------------------------------

climate = pd.read_csv(PRISM_FILE, skiprows=10)
climate.columns = climate.columns.str.strip()

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
    climate["tmean (degrees C)"], errors="coerce"
)
climate = climate.dropna(subset=["Date", "Tmean"]).copy()

climate["Year"] = climate["Date"].dt.year
climate["Month"] = climate["Date"].dt.month
climate["Water_Year"] = climate["Year"]
climate.loc[climate["Month"] >= 10, "Water_Year"] += 1

wy_temperature = (
    climate.groupby("Water_Year")
    .agg(
        Water_Year_Mean_Temperature=("Tmean", "mean"),
        Month_Count=("Tmean", "count")
    )
)
wy_temperature.loc[
    wy_temperature["Month_Count"] != 12,
    "Water_Year_Mean_Temperature"
] = np.nan
wy_temperature.index.name = "Year"
wy_temperature = wy_temperature.reset_index()

# ------------------------------------------------------------
# 3. ACCUMULATION AREA
# ------------------------------------------------------------

acc = pd.read_csv(ACCUMULATION_FILE)
acc.columns = acc.columns.str.strip()

if not {"YEAR", "SUM_Shape_Area"}.issubset(acc.columns):
    raise ValueError(
        "Accumulation CSV needs YEAR and SUM_Shape_Area. "
        f"Found: {acc.columns.tolist()}"
    )

acc["Year"] = pd.to_numeric(acc["YEAR"], errors="coerce")
acc["Accumulation_Area_m2"] = pd.to_numeric(
    acc["SUM_Shape_Area"], errors="coerce"
)
acc = acc.dropna(subset=["Year", "Accumulation_Area_m2"]).copy()
acc["Year"] = acc["Year"].astype(int)
acc = (
    acc.groupby("Year", as_index=False)["Accumulation_Area_m2"]
    .sum()
    .sort_values("Year")
)

acc_temp = acc.merge(
    wy_temperature[["Year", "Water_Year_Mean_Temperature"]],
    on="Year",
    how="left"
)

# ------------------------------------------------------------
# 4. GLACIER EXTENT
# ------------------------------------------------------------

glacier = pd.read_csv(GLACIER_FILE)
glacier.columns = glacier.columns.str.strip()

if not {"Year", "Shape_Area"}.issubset(glacier.columns):
    raise ValueError(
        "Glacier CSV needs Year and Shape_Area. "
        f"Found: {glacier.columns.tolist()}"
    )

glacier["Year"] = pd.to_numeric(glacier["Year"], errors="coerce")
glacier["Glacier_Area_m2"] = pd.to_numeric(
    glacier["Shape_Area"], errors="coerce"
)
glacier = glacier.dropna(subset=["Year", "Glacier_Area_m2"]).copy()
glacier["Year"] = glacier["Year"].astype(int)
glacier = (
    glacier.groupby("Year", as_index=False)["Glacier_Area_m2"]
    .sum()
    .sort_values("Year")
    .drop_duplicates("Year", keep="last")
)

# ------------------------------------------------------------
# 5. INTERPOLATE / EXTRAPOLATE GLACIER AREA AND CALCULATE AAR
# ------------------------------------------------------------

all_years = pd.DataFrame({"Year": np.arange(START_YEAR, END_YEAR + 1)})

aar_df = all_years.merge(
    glacier[["Year", "Glacier_Area_m2"]],
    on="Year",
    how="left"
)

aar_df["Glacier_Area_m2"] = (
    aar_df["Glacier_Area_m2"]
    .interpolate(method="linear", limit_area="inside")
)

first_valid = glacier.iloc[0]
second_valid = glacier.iloc[1]
early_years = aar_df["Year"] < int(first_valid["Year"])

early_slope = (
    second_valid["Glacier_Area_m2"] - first_valid["Glacier_Area_m2"]
) / (second_valid["Year"] - first_valid["Year"])

aar_df.loc[early_years, "Glacier_Area_m2"] = (
    first_valid["Glacier_Area_m2"]
    + (aar_df.loc[early_years, "Year"] - first_valid["Year"]) * early_slope
)

last_valid = glacier.iloc[-1]
second_last_valid = glacier.iloc[-2]
late_years = aar_df["Year"] > int(last_valid["Year"])

late_slope = (
    last_valid["Glacier_Area_m2"] - second_last_valid["Glacier_Area_m2"]
) / (last_valid["Year"] - second_last_valid["Year"])

aar_df.loc[late_years, "Glacier_Area_m2"] = (
    last_valid["Glacier_Area_m2"]
    + (aar_df.loc[late_years, "Year"] - last_valid["Year"]) * late_slope
)

aar_df = aar_df.merge(
    acc[["Year", "Accumulation_Area_m2"]],
    on="Year",
    how="left"
)

aar_df["AAR"] = (
    aar_df["Accumulation_Area_m2"] / aar_df["Glacier_Area_m2"]
)
aar_df.loc[aar_df["Accumulation_Area_m2"].isna(), "AAR"] = np.nan

first_mapped_year = int(glacier["Year"].min())
last_mapped_year = int(glacier["Year"].max())

aar_df["Glacier_Area_Method"] = np.where(
    aar_df["Year"] < first_mapped_year,
    "Extrapolation",
    np.where(
        aar_df["Year"] > last_mapped_year,
        "Extrapolation",
        np.where(
            aar_df["Year"].isin(glacier["Year"]),
            "Mapped",
            "Interpolation"
        )
    )
)

valid_aar = aar_df.dropna(subset=["AAR"]).copy()
aar_slope, aar_intercept = np.polyfit(valid_aar["Year"], valid_aar["AAR"], 1)
aar_df["AAR_Trend"] = aar_slope * aar_df["Year"] + aar_intercept

max_row = valid_aar.loc[valid_aar["AAR"].idxmax()]
min_row = valid_aar.loc[valid_aar["AAR"].idxmin()]

# ------------------------------------------------------------
# 6. LINEAR TRENDS
# ------------------------------------------------------------

snow_trend = linregress(acc["Year"], acc["Accumulation_Area_m2"])
snow_trend_years = np.linspace(acc["Year"].min(), acc["Year"].max(), 200)
snow_trend_area = (
    snow_trend.intercept + snow_trend.slope * snow_trend_years
)

# ------------------------------------------------------------
# 7. PLOT: TWO STACKED PANELS, SHARED X-AXIS
# ------------------------------------------------------------

fig, (ax_area, ax_aar) = plt.subplots(
    2, 1,
    figsize=(16, 14),
    sharex=True,
    gridspec_kw={"height_ratios": [1.65, 1.8]}
)

ax_area.set_title(
    "Variation in Accumulation Area and Glacier Extent Through Time\n"
    "Jefferson Park Glacier",
    fontsize=16,
    fontweight="bold",
    pad=10
)

# The second panel title will be placed at the very bottom of the figure,
# rather than above the lower panel.


# ============================
# TOP PANEL: AREA
# ============================

# Accumulation area line; split at missing 2012
acc_before = acc[acc["Year"] <= 2011]
acc_after = acc[acc["Year"] >= 2013]

ax_area.plot(
    acc_before["Year"], acc_before["Accumulation_Area_m2"],
    color="0.35", linewidth=1.3, alpha=0.85, zorder=2
)
ax_area.plot(
    acc_after["Year"], acc_after["Accumulation_Area_m2"],
    color="0.35", linewidth=1.3, alpha=0.85, zorder=2
)

# Dashed bridge across missing year 2012
before = acc.loc[acc["Year"] == 2011]
after = acc.loc[acc["Year"] == 2013]
if not before.empty and not after.empty:
    ax_area.plot(
        [2011, 2013],
        [
            before["Accumulation_Area_m2"].iloc[0],
            after["Accumulation_Area_m2"].iloc[0]
        ],
        color="#E57373",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=3
    )

# Accumulation-area points coloured by water-year mean temperature
points = acc_temp.dropna(subset=["Water_Year_Mean_Temperature"])
sc = ax_area.scatter(
    points["Year"],
    points["Accumulation_Area_m2"],
    c=points["Water_Year_Mean_Temperature"],
    cmap="coolwarm",
    s=75,
    edgecolors="white",
    linewidths=0.6,
    alpha=0.95,
    zorder=4
)

# Accumulation-area linear trend
ax_area.plot(
    snow_trend_years,
    snow_trend_area,
    color="#4A4A4A",
    linestyle="--",
    linewidth=2.5,
    alpha=0.95,
    zorder=3
)

# Glacier extent observations
ax_area.plot(
    glacier["Year"],
    glacier["Glacier_Area_m2"],
    color="#282828",
    linewidth=2.0,
    marker="x",
    markersize=8,
    markeredgewidth=2.0,
    zorder=3
)

# Label glacier outline observation years
for _, row in glacier.iterrows():
    year = int(row["Year"])
    offset = (0, -12) if year == 1988 else (0, 8)
    va = "top" if year == 1988 else "bottom"

    ax_area.annotate(
        str(year),
        (row["Year"], row["Glacier_Area_m2"]),
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=10
    )

ax_area.set_ylabel("Area (m²)", fontsize=13, fontweight="bold")
ax_area.set_ylim(0, 800000)
ax_area.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ax_area.grid(True, linestyle=":", alpha=0.45)
ax_area.set_axisbelow(True)

# Temperature colorbar for the top panel
cbar = fig.colorbar(sc, ax=ax_area, pad=0.015, shrink=0.85)
cbar.set_label("Water-year average temperature (°C)", fontsize=11)

# Top-panel legend
area_legend = [
    Line2D(
        [0], [0], color="0.65", marker="o",
        markerfacecolor="white", markeredgecolor="0.35",
        markersize=7, linewidth=1.3, label="Accumulation area"
    ),
    Line2D(
        [0], [0], color="#282828", marker="x",
        markersize=8, markeredgewidth=1.5,
        linewidth=1.5, label="Glacier extent"
    ),
    Line2D(
        [0], [0], color="#E57373", linestyle="--",
        linewidth=1.6, label="Missing year (2012)"
    ),
    Line2D(
        [0], [0], color="#4A4A4A", linestyle="--",
        linewidth=2.5, label="Accumulation-area trend"
    ),
]

ax_area.legend(
    handles=area_legend,
    loc="upper right",
    ncol=2,
    frameon=True,
    framealpha=0.95,
    fontsize=9
)

# ============================
# BOTTOM PANEL: AAR
# ============================

# Background zones
ax_aar.axhspan(0, 0.3, color="red", alpha=0.20)
ax_aar.axhspan(0.6, 1.0, color="green", alpha=0.20)
ax_aar.axhline(0.3, color="red", linewidth=1, alpha=0.4)
ax_aar.axhline(0.6, color="green", linewidth=1, alpha=0.4)

# AAR line
ax_aar.plot(
    aar_df["Year"], aar_df["AAR"],
    color="black", marker="o", markersize=4.5,
    linewidth=1.7, label="AAR", zorder=3
)

# Missing AAR years: dashed bridge for isolated missing years
missing_years = aar_df.loc[aar_df["AAR"].isna(), "Year"].tolist()
missing_label_added = False

for year in missing_years:
    previous = aar_df.loc[aar_df["Year"] == year - 1, "AAR"].values
    following = aar_df.loc[aar_df["Year"] == year + 1, "AAR"].values

    if (
        len(previous) > 0 and len(following) > 0
        and not np.isnan(previous[0]) and not np.isnan(following[0])
    ):
        ax_aar.plot(
            [year - 1, year + 1],
            [previous[0], following[0]],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Missing AAR data (2012)" if not missing_label_added else None,
            zorder=2
        )
        missing_label_added = True

# AAR linear trend
ax_aar.plot(
    aar_df["Year"], aar_df["AAR_Trend"],
    color="gray", linestyle="--", linewidth=2.5,
    label="Linear Trend"
)

# Maximum and minimum AAR
ax_aar.scatter(max_row["Year"], max_row["AAR"], color="green", s=100, zorder=5)
ax_aar.annotate(
    f'Max AAR ({int(max_row["Year"])}, {max_row["AAR"]:.2f})',
    xy=(max_row["Year"], max_row["AAR"]),
    xytext=(max_row["Year"] + 1, max_row["AAR"] + 0.10),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10,
    fontweight="bold"
)

ax_aar.scatter(min_row["Year"], min_row["AAR"], color="red", s=100, zorder=5)
ax_aar.annotate(
    f'Min AAR ({int(min_row["Year"])}, {min_row["AAR"]:.2f})',
    xy=(min_row["Year"], min_row["AAR"]),
    xytext=(2021, 0.05),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10,
    fontweight="bold",
    ha="right"
)

# Background labels
ax_aar.text(1986, 0.91, "AAR > 0.6", color="darkgreen", fontsize=11, fontweight="bold")
ax_aar.text(1986, 0.05, "AAR < 0.3", color="darkred", fontsize=11, fontweight="bold")

# Extrapolation shading remains only on the AAR panel
ax_aar.axvspan(
    START_YEAR, first_mapped_year,
    color="lightgray", alpha=0.35, zorder=0
)
ax_aar.axvspan(
    last_mapped_year, END_YEAR,
    color="lightgray", alpha=0.35, zorder=0
)

ax_aar.text(
    (START_YEAR + first_mapped_year) / 2, 0.72,
    "Extrapolation", color="dimgray", fontsize=9,
    ha="center", va="center"
)
ax_aar.text(
    (last_mapped_year + END_YEAR) / 2, 0.72,
    "Extrapolation", color="dimgray", fontsize=9,
    ha="center", va="center"
)

ax_aar.set_ylabel("AAR", fontsize=13, fontweight="bold")
ax_aar.set_ylim(0, 1)
ax_aar.set_yticks(np.arange(0, 1.01, 0.2))
ax_aar.grid(True, linestyle="--", alpha=0.3)
ax_aar.set_axisbelow(True)

ax_aar.legend(
    loc="upper right",
    bbox_to_anchor=(1, 0.97),
    fontsize=10,
    frameon=True
)

# ============================
# SHARED X-AXIS / FORMATTING
# ============================

# Explicitly position the two panels so the shared x-axis is BETWEEN them,
# and leave space for the second title below the lower panel.
fig.subplots_adjust(left=0.08, right=0.91, top=0.92, bottom=0.15, hspace=0)

# Force both panels to have exactly the same horizontal position and width
pos_area = ax_area.get_position()
pos_aar = ax_aar.get_position()

ax_aar.set_position([
    pos_area.x0,
    pos_aar.y0,
    pos_area.width,
    pos_aar.height
])

shared_ticks = np.arange(START_YEAR, END_YEAR + 1, 5)
ax_area.set_xlim(START_YEAR, END_YEAR)
ax_aar.set_xlim(START_YEAR, END_YEAR)
ax_area.set_xticks(shared_ticks)
ax_aar.set_xticks(shared_ticks)

# The upper panel's bottom spine/ticks are the single shared x-axis.
ax_area.spines['bottom'].set_visible(True)
ax_area.tick_params(axis='x', which='major', bottom=True, top=False,
                    labelbottom=True, length=6, direction='in')
ax_area.set_xticks(np.arange(1985, 2027, 5))
ax_area.set_xticklabels(np.arange(1985, 2027, 5))
ax_area.set_zorder(2)
ax_aar.set_zorder(1)
ax_area.patch.set_visible(False)

# Remove all x-axis furniture from the lower panel.
ax_aar.spines['top'].set_visible(False)
ax_aar.spines['bottom'].set_visible(False)
ax_aar.tick_params(axis='x', which='both', bottom=True, top=False,
                    labelbottom=True, labeltop=False)
ax_area.set_xticks(np.arange(1985, 2027, 5))
ax_area.set_xticklabels(np.arange(1985, 2027, 5))

# Shared-axis label at the far right of the middle axis.
ax_area.text(1.01, 0.0, 'Year', transform=ax_area.transAxes,
             ha='left', va='center', fontsize=13, fontweight='bold',
             clip_on=False)

# Panel titles: first at top; second at the very bottom of the figure.
fig.text(0.42, 0.08,
         'Accumulation Area Ratio (AAR), 1985–2026\nJefferson Park Glacier',
         ha='center', va='bottom', fontsize=16, fontweight='bold')

fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# 8. EXPORT COMBINED DATA
# ------------------------------------------------------------

combined = (
    acc.merge(glacier, on="Year", how="outer")
    .merge(
        wy_temperature[["Year", "Water_Year_Mean_Temperature"]],
        on="Year",
        how="outer"
    )
    .merge(
        aar_df[["Year", "AAR", "AAR_Trend", "Glacier_Area_Method"]],
        on="Year",
        how="outer"
    )
    .sort_values("Year")
)

combined.to_csv(OUTPUT_CSV, index=False)

# ------------------------------------------------------------
# 9. PRINT STATISTICS
# ------------------------------------------------------------

print("\n========== AAR STATISTICS ==========")
print(f"Maximum AAR: {max_row['AAR']:.3f} ({int(max_row['Year'])})")
print(f"Minimum AAR: {min_row['AAR']:.3f} ({int(min_row['Year'])})")
print(f"AAR trend slope: {aar_slope:.5f} per year")
print(f"Accumulation-area trend slope: {snow_trend.slope:.2f} m²/year")
print(f"Accumulation-area trend R²: {snow_trend.rvalue ** 2:.4f}")
print(f"Accumulation-area trend p-value: {snow_trend.pvalue:.6f}")
print(f"Missing AAR years: {missing_years}")
print(f"Mapped glacier years: {list(glacier['Year'])}")
print(
    f"Extrapolation periods: {START_YEAR}-{first_mapped_year - 1} "
    f"and {last_mapped_year + 1}-{END_YEAR}"
)

print("\nSaved figure:", OUTPUT_FIGURE)
print("Saved data:", OUTPUT_CSV)
