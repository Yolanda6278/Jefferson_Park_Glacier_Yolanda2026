from pathlib import Path
import pandas as pd
import numpy as np
import os

# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

tas_file = BASE_DIR / "LLNL_downscaled_RCPs_Jeff_Park_Glacier" / "bcsd5" / "tas.csv"

projection_file = BASE_DIR / "LLNL_downscaled_RCPs_Jeff_Park_Glacier" / "bcsd5" / "Projections5.txt"

# PRISM observational temperature data
prism_file = BASE_DIR / "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv"

output_folder = BASE_DIR / "bcsd5_projection_output"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. READ TEMPERATURE DATA
# ============================================================

# CSV has no proper column headers
tas = pd.read_csv(tas_file, header=None)

# First two columns = Year and Month
tas = tas.rename(columns={
    0: "Year",
    1: "Month"
})

# Convert Year and Month to integers
tas["Year"] = tas["Year"].astype(int)
tas["Month"] = tas["Month"].astype(int)

# Temperature columns
temperature_columns = tas.columns[2:]


# ============================================================
# 3. READ PROJECTION NAMES
# ============================================================

with open(projection_file, "r") as f:
    projection_names = [
        line.strip().split("#")[0].strip()
        for line in f
        if line.strip()
    ]

# Check number of projections
assert len(projection_names) == len(temperature_columns), \
    "Number of projection names does not match temperature columns!"

# Rename temperature columns using projection names
tas.columns = ["Year", "Month"] + projection_names

print("Number of projections:", len(projection_names))


# ============================================================
# 4. DEFINE WATER YEAR
# ============================================================

# Water Year:
# October previous year - September current year
#
# Example:
# WY 1991 = October 1990 - September 1991

tas["Water_Year"] = np.where(
    tas["Month"] >= 10,
    tas["Year"] + 1,
    tas["Year"]
)


# ============================================================
# 5. CALCULATE WATER YEAR AVERAGE TEMPERATURE
# ============================================================
tas[projection_names] = tas[projection_names].apply(
    pd.to_numeric, errors="coerce"
)

wya = tas.groupby("Water_Year")[projection_names].mean()

# Only retain complete water years (12 months)
month_count = tas.groupby("Water_Year")["Month"].count()

wya = wya.loc[month_count == 12]

print("\nWYA data:")
print(wya.head())

print("\nWYA period:")
print(wya.index.min(), "-", wya.index.max())


# ============================================================
# 6. CALCULATE 1991-2020 BASELINE FOR EACH PROJECTION
# ============================================================

baseline_start = 1991
baseline_end = 2020

baseline = wya.loc[
    baseline_start:baseline_end
].mean()

print("\nBaseline calculated for:")
print(baseline_start, "-", baseline_end)


# ============================================================
# 7. CALCULATE TEMPERATURE ANOMALIES
# ============================================================

wya_anomaly = wya.subtract(baseline, axis="columns")


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

# WYA absolute temperature
wya.to_csv(
    os.path.join(output_folder, "BCSD5_WYA_temperature.csv")
)

# Baseline temperature for every projection
baseline.to_csv(
    os.path.join(output_folder, "BCSD5_1991-2020_WYA_baseline.csv"),
    header=["Baseline_Temperature"]
)

# WYA temperature anomaly
wya_anomaly.to_csv(
    os.path.join(output_folder, "BCSD5_WYA_temperature_anomaly.csv")
)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print("\nProcessing complete!")

print("\nWYA temperature shape:")
print(wya.shape)

print("\nAnomaly shape:")
print(wya_anomaly.shape)

print("\nExample baseline temperatures:")
print(baseline.head())

print("\nExample anomalies:")
print(wya_anomaly.iloc[:5, :5])

# ============================================================
# 10. READ AND PROCESS PRISM TEMPERATURE DATA
# ============================================================

# PRISM file contains 10 lines of metadata
prism = pd.read_csv(prism_file, skiprows=10)

# Convert Date to datetime
prism["Date"] = pd.to_datetime(prism["Date"], format="%Y-%m")

# Extract Year and Month
prism["Year"] = prism["Date"].dt.year
prism["Month"] = prism["Date"].dt.month

# Convert tmean to numeric
prism["tmean (degrees C)"] = pd.to_numeric(
    prism["tmean (degrees C)"],
    errors="coerce"
)

# Define Water Year
# October previous year - September current year

prism["Water_Year"] = np.where(
    prism["Month"] >= 10,
    prism["Year"] + 1,
    prism["Year"]
)

# Calculate Water Year Average Temperature
prism_wya = prism.groupby("Water_Year")["tmean (degrees C)"].mean()

# Only retain complete Water Years
prism_month_count = prism.groupby("Water_Year")["Month"].count()

prism_wya = prism_wya.loc[
    prism_month_count == 12
]

print("\nPRISM WYA period:")
print(prism_wya.index.min(), "-", prism_wya.index.max())


# ============================================================
# 11. CALCULATE PRISM 1991-2020 BASELINE AND ANOMALY
# ============================================================

baseline_start = 1991
baseline_end = 2020

prism_baseline = prism_wya.loc[
    baseline_start:baseline_end
].mean()

prism_anomaly = prism_wya - prism_baseline

print("\nPRISM baseline:")
print(prism_baseline)

print("\nPRISM anomaly mean during 1991-2020:")
print(prism_anomaly.loc[1991:2020].mean())


# Save PRISM outputs

prism_wya.to_csv(
    os.path.join(output_folder, "PRISM_WYA_temperature.csv"),
    header=["WYA_Temperature"]
)

pd.Series(
    [prism_baseline],
    index=["1991-2020"]
).to_csv(
    os.path.join(output_folder, "PRISM_1991-2020_WYA_baseline.csv"),
    header=["Baseline_Temperature"]
)

prism_anomaly.to_csv(
    os.path.join(output_folder, "PRISM_WYA_temperature_anomaly.csv"),
    header=["Temperature_Anomaly"]
)

# ============================================================
# 11B. CALIBRATE BCSD5 TO PRISM 1991-2020 ABSOLUTE BASELINE
# ============================================================

# Calculate historical baseline temperature for each BCSD5 projection
bcsd5_baseline = wya.loc[
    baseline_start:baseline_end
].mean()

# Calculate bias relative to PRISM baseline
bcsd5_bias = bcsd5_baseline - prism_baseline

# Correct each projection using its own historical bias
wya_calibrated = wya.subtract(bcsd5_bias, axis="columns")

# Calculate temperature anomaly relative to PRISM baseline
wya_calibrated_anomaly = wya_calibrated - prism_baseline

print("\nBCSD5 calibration to PRISM:")
print("PRISM baseline:", prism_baseline)

print("\nMean calibrated BCSD5 baseline:")
print(
    wya_calibrated.loc[
        baseline_start:baseline_end
    ].mean().mean()
)

print("\nMean calibrated BCSD5 anomaly during 1991-2020:")
print(
    wya_calibrated_anomaly.loc[
        baseline_start:baseline_end
    ].mean().mean()
)

# Save calibrated BCSD5 temperature

wya_calibrated.to_csv(
    os.path.join(
        output_folder,
        "BCSD5_WYA_temperature_PRISM_calibrated.csv"
    )
)

wya_calibrated_anomaly.to_csv(
    os.path.join(
        output_folder,
        "BCSD5_WYA_temperature_anomaly_PRISM_baseline.csv"
    )
)

bcsd5_bias.to_csv(
    os.path.join(
        output_folder,
        "BCSD5_PRISM_baseline_bias.csv"
    ),
    header=["Temperature_Bias"]
)

# ============================================================
# CHECK BASELINE ALIGNMENT
# ============================================================

# PRISM 1991-2020 absolute temperature baseline
print("PRISM baseline:")
print(prism_baseline)

# Mean calibrated BCSD5 temperature during 1991-2020
calibrated_baseline = wya_calibrated.loc[1991:2020].mean()

print("\nMean calibrated BCSD5 baseline:")
print(calibrated_baseline.mean())

print("\nDifference from PRISM baseline:")
print(calibrated_baseline - prism_baseline)

print("\nMean difference across all projections:")
print((calibrated_baseline - prism_baseline).mean())

print("\nMaximum absolute difference:")
print((calibrated_baseline - prism_baseline).abs().max())

# ============================================================
# 12. READ BCSD5 ANOMALY DATA AND CALCULATE ENSEMBLE STATISTICS
# ============================================================

bcsd5_anomaly_file = os.path.join(
    output_folder,
    "BCSD5_WYA_temperature_anomaly.csv"
)

bcsd5_anomaly = pd.read_csv(
    bcsd5_anomaly_file,
    index_col=0
)

# Convert Water Year index to integer
bcsd5_anomaly.index = bcsd5_anomaly.index.astype(int)

# Calculate ensemble statistics for each year

bcsd5_ensemble_mean = bcsd5_anomaly.mean(axis=1)

bcsd5_ensemble_5th = bcsd5_anomaly.quantile(
    0.05,
    axis=1
)

bcsd5_ensemble_95th = bcsd5_anomaly.quantile(
    0.95,
    axis=1
)

bcsd5_ensemble_median = bcsd5_anomaly.median(axis=1)


# ============================================================
# 13. SELECT COMMON HISTORICAL PERIOD
# ============================================================

# PRISM and BCSD5 must have the same years for comparison

common_years = prism_anomaly.index.intersection(
    bcsd5_anomaly.index
)

# Restrict comparison to historical PRISM period
common_years = common_years[
    (common_years >= prism_anomaly.index.min()) &
    (common_years <= prism_anomaly.index.max())
]

prism_comparison = prism_anomaly.loc[common_years]

bcsd5_mean_comparison = bcsd5_ensemble_mean.loc[common_years]

bcsd5_5th_comparison = bcsd5_ensemble_5th.loc[common_years]

bcsd5_95th_comparison = bcsd5_ensemble_95th.loc[common_years]


print("\nHistorical comparison period:")
print(common_years.min(), "-", common_years.max())

print("\nNumber of comparison years:")
print(len(common_years))


# ============================================================
# 14. PLOT PRISM VS BCSD5 HISTORICAL TEMPERATURE ANOMALIES
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# BCSD5 ensemble spread
plt.fill_between(
    common_years,
    bcsd5_5th_comparison,
    bcsd5_95th_comparison,
    color="lightgray",
    alpha=0.6,
    label="BCSD5 5th–95th percentile"
)

# BCSD5 ensemble mean
plt.plot(
    common_years,
    bcsd5_mean_comparison,
    color="black",
    linewidth=2,
    label="BCSD5 ensemble mean"
)

# PRISM observation
plt.plot(
    common_years,
    prism_comparison,
    color="#e65715",
    linewidth=2,
    label="PRISM observation"
)

plt.axhline(
    0,
    color="gray",
    linestyle="--",
    linewidth=1
)

plt.xlabel("Water Year", fontsize=12)

plt.ylabel(
    "Temperature Anomaly (°C)",
    fontsize=12
)

plt.title(
    "PRISM vs BCSD5 Historical Temperature Anomalies",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "PRISM_vs_BCSD5_Historical_Anomaly.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 15. CALCULATE HISTORICAL STATISTICS
# ============================================================

from scipy.stats import linregress

# ------------------------------------------------------------
# PRISM statistics
# ------------------------------------------------------------

prism_mean = prism_comparison.mean()

prism_std = prism_comparison.std()

prism_trend = linregress(
    common_years,
    prism_comparison
)

prism_trend_decade = prism_trend.slope * 10


# ------------------------------------------------------------
# BCSD5 ensemble mean statistics
# ------------------------------------------------------------

bcsd5_mean = bcsd5_mean_comparison.mean()

bcsd5_std = bcsd5_mean_comparison.std()

bcsd5_trend = linregress(
    common_years,
    bcsd5_mean_comparison
)

bcsd5_trend_decade = bcsd5_trend.slope * 10


# ------------------------------------------------------------
# Comparison statistics
# ------------------------------------------------------------

correlation = np.corrcoef(
    prism_comparison,
    bcsd5_mean_comparison
)[0, 1]

rmse = np.sqrt(
    np.mean(
        (prism_comparison - bcsd5_mean_comparison) ** 2
    )
)

bias = np.mean(
    bcsd5_mean_comparison - prism_comparison
)


# ============================================================
# 16. CALCULATE TREND DISTRIBUTION OF INDIVIDUAL PROJECTIONS
# ============================================================

projection_trends = []

for projection in bcsd5_anomaly.columns:

    projection_data = bcsd5_anomaly.loc[common_years, projection]

    trend = linregress(
        common_years,
        projection_data
    )

    projection_trends.append(
        trend.slope * 10
    )

projection_trends = np.array(projection_trends)

# ============================================================
# CALCULATE BCSD5 ENSEMBLE COVERAGE
# ============================================================

prism_within_ensemble = (
    (prism_comparison >= bcsd5_5th_comparison) &
    (prism_comparison <= bcsd5_95th_comparison)
)

coverage_percentage = (
    prism_within_ensemble.sum() /
    len(prism_within_ensemble)
) * 100

print("\nPRISM within BCSD5 5th-95th percentile range:")
print(f"{coverage_percentage:.2f}%")

# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)

print("HISTORICAL TEMPERATURE COMPARISON")
print("=" * 60)

print("\nPRISM:")
print(f"Mean anomaly: {prism_mean:.3f} °C")
print(f"Standard deviation: {prism_std:.3f} °C")
print(f"Trend: {prism_trend_decade:.3f} °C/decade")
print(f"Trend p-value: {prism_trend.pvalue:.4f}")

print("\nBCSD5 Ensemble Mean:")
print(f"Mean anomaly: {bcsd5_mean:.3f} °C")
print(f"Standard deviation: {bcsd5_std:.3f} °C")
print(f"Trend: {bcsd5_trend_decade:.3f} °C/decade")
print(f"Trend p-value: {bcsd5_trend.pvalue:.4f}")

print("\nComparison:")
print(f"Correlation: {correlation:.3f}")
print(f"RMSE: {rmse:.3f} °C")
print(f"Bias (BCSD5 - PRISM): {bias:.3f} °C")

print("\nIndividual BCSD5 Projection Trends:")
print(f"Mean trend: {projection_trends.mean():.3f} °C/decade")
print(f"Median trend: {np.median(projection_trends):.3f} °C/decade")
print(f"5th percentile: {np.percentile(projection_trends, 5):.3f} °C/decade")
print(f"95th percentile: {np.percentile(projection_trends, 95):.3f} °C/decade")
print(f"Minimum trend: {projection_trends.min():.3f} °C/decade")
print(f"Maximum trend: {projection_trends.max():.3f} °C/decade")


# ============================================================
# 18. SAVE HISTORICAL COMPARISON STATISTICS
# ============================================================

comparison_statistics = pd.DataFrame({
    "Statistic": [
        "Mean anomaly",
        "Standard deviation",
        "Linear trend (°C/decade)",
        "Trend p-value",
        "Correlation with PRISM",
        "RMSE",
        "Bias (BCSD5 - PRISM)"
    ],

    "PRISM": [
        prism_mean,
        prism_std,
        prism_trend_decade,
        prism_trend.pvalue,
        np.nan,
        np.nan,
        np.nan
    ],

    "BCSD5 Ensemble Mean": [
        bcsd5_mean,
        bcsd5_std,
        bcsd5_trend_decade,
        bcsd5_trend.pvalue,
        correlation,
        rmse,
        bias
    ]
})

comparison_statistics.to_csv(
    os.path.join(
        output_folder,
        "PRISM_vs_BCSD5_Historical_Statistics.csv"
    ),
    index=False
)


# Save individual projection trend results

projection_trend_df = pd.DataFrame({
    "Projection": bcsd5_anomaly.columns,
    "Trend (°C/decade)": projection_trends
})

projection_trend_df.to_csv(
    os.path.join(
        output_folder,
        "BCSD5_Historical_Projection_Trends.csv"
    ),
    index=False
)


print("\nHistorical comparison completed!")

# ============================================================
# 19. FUTURE TEMPERATURE THRESHOLD EXCEEDANCE
# ============================================================

# Temperature thresholds derived from historical PRISM-AAR relationship

thresholds = {
    "Low threshold (-0.466°C)": -0.466,
    "Primary threshold (+0.351°C)": 0.351,
    "High threshold (+1.168°C)": 1.168
}

# Future period
future_start = 2021
future_end = 2099

projection_anomaly = wya_calibrated_anomaly.loc[
    1980:2099
]

print("\nFuture projection period:")
print(projection_anomaly.index.min(), "-", projection_anomaly.index.max())

print("\nNumber of projections:")
print(projection_anomaly.shape[1])

# ------------------------------------------------------------
# Calculate annual exceedance percentage
# ------------------------------------------------------------

exceedance_percentage = pd.DataFrame(
    index=projection_anomaly.index
)

for threshold_name, threshold_value in thresholds.items():

    exceedance_percentage[threshold_name] = (
        ((projection_anomaly > threshold_value).sum(axis=1))
        / projection_anomaly.shape[1]
        * 100
    )

print("\nAnnual exceedance percentage:")
print(exceedance_percentage.head())

# ------------------------------------------------------------
# Calculate 7-year moving average
# ------------------------------------------------------------

exceedance_percentage_7yr = exceedance_percentage.rolling(
    window=7,
    center=True
).mean()


# ------------------------------------------------------------
# Save exceedance percentages
# ------------------------------------------------------------

exceedance_percentage.to_csv(
    os.path.join(
        output_folder,
        "BCSD5_Annual_Temperature_Threshold_Exceedance.csv"
    )
)

exceedance_percentage_7yr.to_csv(
    os.path.join(
        output_folder,
        "BCSD5_7yr_Moving_Average_Threshold_Exceedance.csv"
    )
)


# ============================================================
# 20. PLOT TEMPERATURE THRESHOLD EXCEEDANCE BY RCP
# ============================================================

# ------------------------------------------------------------
# Define thresholds
# ------------------------------------------------------------

thresholds = {
    "-0.466°C": -0.466334,
    "+0.351°C": 0.350791,
    "+1.168°C": 1.167915
}

threshold_colors = {
    "-0.466°C": "#287C8E",
    "+0.351°C": "#E5B04A",
    "+1.168°C": "#D18445"
}

# ------------------------------------------------------------
# Identify RCP for each projection
# ------------------------------------------------------------

rcp_groups = {
    "RCP 2.6": [],
    "RCP 4.5": [],
    "RCP 6.0": [],
    "RCP 8.5": []
}

for projection in projection_names:

    if projection.endswith("rcp26"):
        rcp_groups["RCP 2.6"].append(projection)

    elif projection.endswith("rcp45"):
        rcp_groups["RCP 4.5"].append(projection)

    elif projection.endswith("rcp60"):
        rcp_groups["RCP 6.0"].append(projection)

    elif projection.endswith("rcp85"):
        rcp_groups["RCP 8.5"].append(projection)


# ------------------------------------------------------------
# Print number of projections in each RCP
# ------------------------------------------------------------

print("\nNumber of projections in each RCP:")

for rcp_name, models in rcp_groups.items():
    print(f"{rcp_name}: {len(models)}")


# ------------------------------------------------------------
# Create 2 x 2 plots
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

rcp_order = [
    "RCP 2.6",
    "RCP 4.5",
    "RCP 6.0",
    "RCP 8.5"
]


# ------------------------------------------------------------
# Plot each RCP
# ------------------------------------------------------------

for ax, rcp_name in zip(axes, rcp_order):

    models = rcp_groups[rcp_name]

    n_models = len(models)

    print(f"\nProcessing {rcp_name}: {n_models} projections")

    # Get temperature data for this RCP
    rcp_temperature = wya_calibrated_anomaly[models]

    # Calculate percentage exceeding each threshold
    rcp_exceedance = pd.DataFrame(
        index=rcp_temperature.index
    )

    for threshold_name, threshold_value in thresholds.items():

        rcp_exceedance[threshold_name] = (
            (rcp_temperature > threshold_value).sum(axis=1)
            / n_models
            * 100
        )

    # --------------------------------------------------------
    # Plot each threshold
    # --------------------------------------------------------

    for threshold_name in thresholds.keys():

        ax.plot(
            rcp_exceedance.index,
            rcp_exceedance[threshold_name],
            color=threshold_colors[threshold_name],
            linewidth=1.8,
            label=threshold_name
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        f"{rcp_name} Annual Temperature Average",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlim(1980, 2099)

    ax.set_ylim(0, 100)

    ax.set_xticks(
        np.arange(1980, 2100, 20)
    )

    ax.set_yticks(
        np.arange(0, 101, 20)
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        loc="upper left",
        fontsize=10,
        title="Temperature threshold"
    )


# ------------------------------------------------------------
# Shared axis labels
# ------------------------------------------------------------

fig.supxlabel(
    "Water Year",
    fontsize=13
)

fig.supylabel(
    "Percentage of Models Exceeding Threshold (%)",
    fontsize=13
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "BCSD5_RCP_Temperature_Threshold_Exceedance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 21. PLOT PRIMARY THRESHOLD
# ============================================================

plt.figure(figsize=(14, 7))

# ------------------------------------------------------------
# Background shading
# ------------------------------------------------------------

plt.axvspan(
    2021, 2040,
    color="red",
    alpha=0.15
)

plt.axvspan(
    2040, 2060,
    color="red",
    alpha=0.25
)

plt.axvspan(
    2060, 2080,
    color="red",
    alpha=0.35
)

plt.axvspan(
    2080, 2099,
    color="red",
    alpha=0.45
)


# ------------------------------------------------------------
# Annual percentage
# ------------------------------------------------------------

plt.plot(
    exceedance_percentage.index,
    exceedance_percentage["Primary threshold (+0.351°C)"],
    color="lightgray",
    linewidth=1,
    label="Annual percentage"
)


# ------------------------------------------------------------
# 7-year moving average
# ------------------------------------------------------------

plt.plot(
    exceedance_percentage_7yr.index,
    exceedance_percentage_7yr["Primary threshold (+0.351°C)"],
    color="green",
    linewidth=2.5,
    label="7-year moving average"
)


# ------------------------------------------------------------
# Mark first year reaching 60%
# ------------------------------------------------------------

target_percentage = 60

series = exceedance_percentage_7yr[
    "Primary threshold (+0.351°C)"
]

crossing_years = series[
    series >= target_percentage
].index

if len(crossing_years) > 0:

    first_year = crossing_years[0]

    first_percentage = series.loc[first_year]

    plt.scatter(
        first_year,
        first_percentage,
        color="red",
        s=80,
        edgecolor="black",
        zorder=5
    )

    plt.annotate(
        f"{first_year}",
        xy=(first_year, first_percentage),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

    print(
        f"Primary threshold first reaches "
        f"{target_percentage}% in {first_year}"
    )


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

plt.xlabel("Water Year", fontsize=12)

plt.ylabel(
    "Projections Exceeding +0.351°C (%)",
    fontsize=12
)

plt.title(
    "Future Exceedance of Primary Temperature Threshold",
    fontsize=16,
    fontweight="bold"
)

plt.xlim(1980, 2099)

plt.ylim(0, 100)

plt.xticks(
    np.arange(1980, 2100, 10)
)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "BCSD5_Primary_Threshold_Exceedance.png"
    ),
    dpi=300
)

plt.show()

# ============================================================
# 22. PLOT RCP THRESHOLD EXCEEDANCE WITH 60% CROSSING YEARS
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

rcp_order = [
    "RCP 2.6",
    "RCP 4.5",
    "RCP 6.0",
    "RCP 8.5"
]

target_percentage = 60

required_consecutive_years = 5


for ax, rcp_name in zip(axes, rcp_order):

    models = rcp_groups[rcp_name]

    n_models = len(models)

    # --------------------------------------------------------
    # Temperature data for this RCP
    # --------------------------------------------------------

    rcp_temperature = wya_calibrated_anomaly[models]

    # --------------------------------------------------------
    # Calculate annual exceedance percentages
    # --------------------------------------------------------

    rcp_exceedance = pd.DataFrame(
        index=rcp_temperature.index
    )

    for threshold_name, threshold_value in thresholds.items():

        rcp_exceedance[threshold_name] = (
            (rcp_temperature > threshold_value).sum(axis=1)
            / n_models
            * 100
        )

    # --------------------------------------------------------
    # Plot annual percentage curves
    # --------------------------------------------------------

    for threshold_name in thresholds.keys():

        series = rcp_exceedance[threshold_name]

        ax.plot(
            series.index,
            series,
            color=threshold_colors[threshold_name],
            linewidth=1.8,
            label=threshold_name
        )

        # ----------------------------------------------------
        # Find first year reaching 60% and remaining >= 60%
        # for at least 5 consecutive years
        # ----------------------------------------------------

        crossing_year = None

        for year in series.index:

            if series.loc[year] >= target_percentage:

                consecutive_years = series.loc[
                    year:year + required_consecutive_years - 1
                ]

                if (
                    len(consecutive_years) == required_consecutive_years
                    and (consecutive_years >= target_percentage).all()
                ):

                    crossing_year = year
                    break

        # ----------------------------------------------------
        # Special case:
        # Do not mark the lowest curve in RCP 2.6
        # ----------------------------------------------------

        skip_crossing = (
            rcp_name == "RCP 2.6"
            and threshold_name == list(thresholds.keys())[-1]
        )

        # ----------------------------------------------------
        # Mark crossing point
        # ----------------------------------------------------

        if crossing_year is not None and not skip_crossing:

            ax.scatter(
                crossing_year,
                target_percentage,
                color=threshold_colors[threshold_name],
                edgecolor="black",
                s=65,
                zorder=5
            )

            ax.annotate(
                str(crossing_year),
                xy=(crossing_year, target_percentage),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="black"
            )

            print(
                f"{rcp_name} | {threshold_name} | "
                f"Stable crossing year: {crossing_year}"
            )

        elif skip_crossing:

            print(
                f"{rcp_name} | {threshold_name} | "
                f"Crossing annotation skipped"
            )

        else:

            print(
                f"{rcp_name} | {threshold_name} | "
                f"Never reaches stable {target_percentage}%"
            )

    # --------------------------------------------------------
    # 60% reference line
    # --------------------------------------------------------

    ax.axhline(
        target_percentage,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        f"{rcp_name} Models Exceeding Temperature Thresholds",
        fontsize=14,
        fontweight="bold"
    )

    # Leave empty space before 1980 and after 2100
    ax.set_xlim(1970, 2110)

    ax.set_ylim(-5, 105)

    ax.set_xticks(
        np.arange(1980, 2101, 20)
    )

    ax.set_yticks(
        np.arange(0, 101, 20)
    )

    ax.grid(alpha=0.3)

    ax.legend(
        loc="upper left",
        fontsize=9,
        title="Temperature threshold"
    )


# ------------------------------------------------------------
# Shared axis labels
# ------------------------------------------------------------

fig.supxlabel(
    "Water Year",
    fontsize=13
)

fig.supylabel(
    "Percentage of Models Exceeding Threshold (%)",
    fontsize=13
)


# ------------------------------------------------------------
# Layout and save
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "BCSD5_RCP_Temperature_Threshold_Exceedance_60percent.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# # ============================================================
# # 23. PLOT PERCENTAGE OF MODELS WITH AAR < 0.3
# # BASED ON TEMPERATURE THRESHOLDS
# # ============================================================

# fig, axes = plt.subplots(
#     2, 2,
#     figsize=(15, 10),
#     sharex=True,
#     sharey=True
# )

# axes = axes.flatten()

# target_percentage = 60
# required_consecutive_years = 5

# rcp_order = [
#     "RCP 2.6",
#     "RCP 4.5",
#     "RCP 6.0",
#     "RCP 8.5"
# ]

# for ax, rcp_name in zip(axes, rcp_order):

#     models = rcp_groups[rcp_name]

#     n_models = len(models)

#     # --------------------------------------------------------
#     # Temperature data for this RCP
#     # --------------------------------------------------------

#     rcp_temperature = wya_calibrated_anomaly[models]

#     # --------------------------------------------------------
#     # Calculate percentage of models associated with AAR < 0.3
#     # based on the temperature thresholds
#     # --------------------------------------------------------

#     rcp_aar_percentage = pd.DataFrame(
#         index=rcp_temperature.index
#     )

#     for threshold_name, threshold_value in thresholds.items():

#         rcp_aar_percentage[threshold_name] = (
#             (rcp_temperature > threshold_value).sum(axis=1)
#             / n_models
#             * 100
#         )

#     # --------------------------------------------------------
#     # Plot percentage curves
#     # --------------------------------------------------------

#     for threshold_name in thresholds.keys():

#         # --------------------------------------------------------
#         # Add AAR < 0.3 line
#         # Based on the primary temperature threshold (+0.350791°C)
#         # --------------------------------------------------------

#         aar_threshold = 0.350791

#         aar_percentage = (
#             (rcp_temperature > aar_threshold).sum(axis=1)
#             / n_models
#             * 100
#         )

#         ax.plot(
#             aar_percentage.index,
#             aar_percentage,
#             color="black",
#             linewidth=2.5,
#             linestyle="-",
#             label="AAR < 0.3"
#         )

# # --------------------------------------------------------
# # Find first year reaching 60% and remaining >=60%
# # for 5 consecutive years
# # --------------------------------------------------------

#         aar_crossing_year = None

#         for year in aar_percentage.index:

#             if aar_percentage.loc[year] >= target_percentage:

#                 consecutive_years = aar_percentage.loc[
#                     year:year + required_consecutive_years - 1
#                 ]

#                 if (
#                     len(consecutive_years) == required_consecutive_years
#                     and (consecutive_years >= target_percentage).all()
#                 ):

#                     aar_crossing_year = year
#                     break

# # --------------------------------------------------------
# # Mark AAR < 0.3 crossing year
# # --------------------------------------------------------

#         if aar_crossing_year is not None:

#             ax.scatter(
#                 aar_crossing_year,
#                 target_percentage,
#                 color="black",
#                 marker="o",
#                 s=65,
#                 zorder=6
#             )

#             ax.annotate(
#                 str(aar_crossing_year),
#                 xy=(aar_crossing_year, target_percentage),
#                 xytext=(0, -18),
#                 textcoords="offset points",
#                 ha="center",
#                 va="top",
#                 fontsize=9,
#                 color="black",
#                 fontweight="bold"
#             )

#             print(
#                 f"{rcp_name} | AAR < 0.3 | "
#                 f"Stable crossing year: {aar_crossing_year}"
#             )

#         series = rcp_aar_percentage[threshold_name]

#         ax.plot(
#             series.index,
#             series,
#             color=threshold_colors[threshold_name],
#             linewidth=1.8,
#             label=threshold_name
#         )

#         # ----------------------------------------------------
#         # Find first year reaching 60% and remaining >=60%
#         # for 5 consecutive years
#         # ----------------------------------------------------

#         crossing_year = None

#         for year in series.index:

#             if series.loc[year] >= target_percentage:

#                 consecutive_years = series.loc[
#                     year:year + required_consecutive_years - 1
#                 ]

#                 if (
#                     len(consecutive_years)
#                     == required_consecutive_years
#                     and
#                     (consecutive_years >= target_percentage).all()
#                 ):

#                     crossing_year = year
#                     break

#         # ----------------------------------------------------
#         # Skip crossing marker for lowest threshold curve
#         # in RCP 2.6
#         # ----------------------------------------------------

#         skip_crossing = (
#             rcp_name == "RCP 2.6"
#             and threshold_name == list(thresholds.keys())[-1]
#         )

#         # ----------------------------------------------------
#         # Plot crossing marker and year annotation
#         # ----------------------------------------------------

#         if crossing_year is not None and not skip_crossing:

#             ax.scatter(
#                 crossing_year,
#                 target_percentage,
#                 color=threshold_colors[threshold_name],
#                 marker="o",
#                 s=45,
#                 zorder=5
#             )

#             ax.annotate(
#                 str(crossing_year),
#                 xy=(crossing_year, target_percentage),
#                 xytext=(0, 8),
#                 textcoords="offset points",
#                 ha="center",
#                 va="bottom",
#                 fontsize=9,
#                 color=threshold_colors[threshold_name]
#             )

#             print(
#                 f"{rcp_name} | {threshold_name} | "
#                 f"First stable year reaching "
#                 f"{target_percentage}% of models with AAR < 0.3: "
#                 f"{crossing_year}"
#             )

#         elif skip_crossing:

#             print(
#                 f"{rcp_name} | {threshold_name} | "
#                 f"Crossing annotation skipped"
#             )

#         else:

#             print(
#                 f"{rcp_name} | {threshold_name} | "
#                 f"Never reaches stable {target_percentage}%"
#             )

#     # --------------------------------------------------------
#     # 60% reference line
#     # --------------------------------------------------------

#     ax.axhline(
#         target_percentage,
#         color="black",
#         linestyle="--",
#         linewidth=1.2,
#         alpha=0.8
#     )

#     # --------------------------------------------------------
#     # Axes formatting
#     # --------------------------------------------------------

#     ax.set_title(
#         rcp_name,
#         fontsize=13,
#         fontweight="bold"
#     )

#     ax.set_xlim(1970, 2110)

#     ax.set_ylim(-5, 105)

#     ax.set_xticks(
#         np.arange(1980, 2101, 20)
#     )

#     ax.set_yticks(
#         np.arange(0, 101, 20)
#     )

#     ax.grid(
#         True,
#         linestyle=":",
#         alpha=0.4
#     )

#     ax.legend(
#         loc="lower right",
#         fontsize=9,
#         title="Temperature threshold"
#     )

# # ------------------------------------------------------------
# # Shared axis labels
# # ------------------------------------------------------------

# fig.supxlabel(
#     "Water Year",
#     fontsize=12
# )

# fig.supylabel(
#     "Percentage of climate models associated with AAR < 0.3 (%)",
#     fontsize=12
# )

# fig.suptitle(
#     "Percentage of Climate Models Projected to Have AAR < 0.3",
#     fontsize=16,
#     fontweight="bold",
#     y=0.98
# )

# # ------------------------------------------------------------
# # Layout
# # ------------------------------------------------------------

# plt.tight_layout(
#     rect=[0.03, 0.03, 1, 0.95]
# )

# # ------------------------------------------------------------
# # Save figure
# # ------------------------------------------------------------

# plt.savefig(
#     os.path.join(
#         output_folder,
#         "Section23_Percentage_Models_AAR_below_0.3.png"
#     ),
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.savefig(
#     os.path.join(
#         output_folder,
#         "Section23_Percentage_Models_AAR_below_0.3.pdf"
#     ),
#     bbox_inches="tight"
# )

# plt.show()