# ============================================================
# NOAA RECTANGLE vs OBSERVATIONAL DATA COMPARISON
# Period: 1950-1999
#
# Comparisons:
# 1. May-October
# 2. November-October
# 3. Water Year (October-September)
#
# Outputs:
# - Time-series plots
# - Trend comparison plots
# - Scatter plots with 1:1 line
# - Difference plots
# - Statistical summary CSV
# - Matched climate-period data CSVs
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.formula.api as smf

from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

# Observation files
OBS_MEAN_FILE = BASE_DIR / "1_8obs_regional" / "1_8obs" / "Tavg_SpatialStat_mean.csv"
OBS_STDEV_FILE = BASE_DIR / "1_8obs_regional" / "1_8obs" / "Tavg_SpatialStat_stdev.csv"

# NOAA regional rectangle datasets
NOAA_FILES = {
    "May-October": "May_October_6month_temp_ave_1895_2026(F).txt",

    "November-October":
        "northern_cascades_Nov_Oct_12month_temp_1895_2026(F).txt",

    "Water Year":
        "Oct_Sep_12month_temp(F)_1895_2026.txt",
}

# Output directory
OUTPUT_DIR = BASE_DIR / "output_NOAA_observation_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. GENERAL SETTINGS
# ============================================================

START_YEAR = 1950
END_YEAR = 1999

BASELINE_START = 1950
BASELINE_END = 1999

# NOAA temperature is Fahrenheit
# Observation temperature is assumed to be Celsius


# ============================================================
# 3. READ OBSERVATIONAL DATA
# ============================================================

obs_mean = pd.read_csv(
    OBS_MEAN_FILE,
    header=None,
    names=["Year", "Month", "Temperature"]
)

obs_stdev = pd.read_csv(
    OBS_STDEV_FILE,
    header=None,
    names=["Year", "Month", "Spatial_Stdev"]
)

# Ensure numeric
obs_mean["Year"] = pd.to_numeric(obs_mean["Year"])
obs_mean["Month"] = pd.to_numeric(obs_mean["Month"])
obs_mean["Temperature"] = pd.to_numeric(obs_mean["Temperature"])

obs_stdev["Year"] = pd.to_numeric(obs_stdev["Year"])
obs_stdev["Month"] = pd.to_numeric(obs_stdev["Month"])
obs_stdev["Spatial_Stdev"] = pd.to_numeric(
    obs_stdev["Spatial_Stdev"]
)

# Keep October-December of the previous year so Water Year 1950
# (October 1949-September 1950) can be calculated completely.
obs_mean = obs_mean[
    (obs_mean["Year"] >= START_YEAR - 1) &
    (obs_mean["Year"] <= END_YEAR)
].copy()

obs_stdev = obs_stdev[
    (obs_stdev["Year"] >= START_YEAR) &
    (obs_stdev["Year"] <= END_YEAR)
].copy()


# ============================================================
# 4. READ NOAA DATA
# ============================================================

def read_noaa_file(filepath):

    """
    NOAA files contain metadata/comment lines followed by:
    Date,Value

    Date format:
    YYYYMM

    Value:
    Temperature in Fahrenheit
    """

    df = pd.read_csv(
        filepath,
        skiprows=3,
        sep=",",
        header=None
    )

    # Remove potential header row
    df = df.iloc[:, :2].copy()

    df.columns = ["Date", "Value"]

    # Convert Date to string
    df["Date"] = df["Date"].astype(str).str.strip()

    # Extract year
    df["Year"] = df["Date"].str[:4].astype(int)

    # Convert Fahrenheit to Celsius
    df["Temperature_C"] = (
        df["Value"] - 32
    ) / 1.8

    # Restrict years
    df = df[
        (df["Year"] >= START_YEAR) &
        (df["Year"] <= END_YEAR)
    ].copy()

    return df


# ============================================================
# 5. CALCULATE OBSERVATIONAL PERIOD MEANS
# ============================================================

def calculate_observation_period(
    data,
    period_name
):

    """
    Calculate seasonal mean temperature
    from monthly observational data.
    """

    if period_name == "May-October":

        months = [5, 6, 7, 8, 9, 10]

        result = (
            data[data["Month"].isin(months)]
            .groupby("Year")["Temperature"]
            .mean()
        )

    elif period_name == "November-October":

        # November-December belong to previous year
        # January-October belong to current year

        temp = data.copy()

        temp["ClimateYear"] = temp["Year"]

        temp.loc[
            temp["Month"].isin([11, 12]),
            "ClimateYear"
        ] += 1

        months = list(range(1, 11)) + [11, 12]

        result = (
            temp[temp["Month"].isin(months)]
            .groupby("ClimateYear")["Temperature"]
            .mean()
        )

        result.index.name = "Year"

    elif period_name == "Water Year":

        # October-December belong to previous year
        # January-September belong to current year

        temp = data.copy()

        temp["ClimateYear"] = temp["Year"]

        temp.loc[
            temp["Month"].isin([10, 11, 12]),
            "ClimateYear"
        ] += 1

        months = list(range(1, 10)) + [10, 11, 12]

        wy_data = temp[temp["Month"].isin(months)]

        # Require all 12 months for a complete Water Year
        grouped = wy_data.groupby("ClimateYear")["Temperature"]
        result = grouped.mean()
        month_count = grouped.count()
        result = result[month_count == 12]

        result.index.name = "Year"

    else:

        raise ValueError(
            f"Unknown period: {period_name}"
        )

    result = result.reset_index()

    result.columns = [
        "Year",
        "Observation_C"
    ]

    return result


# ============================================================
# 6. CALCULATE NOAA PERIODS
# ============================================================

def calculate_noaa_period(
    noaa_data,
    period_name
):

    """
    NOAA data already represent the selected
    seasonal/annual period.

    Convert temperature to Celsius and return
    annual values.
    """

    result = noaa_data[
        ["Year", "Temperature_C"]
    ].copy()

    result.columns = [
        "Year",
        "Rectangle_C"
    ]

    return result


# ============================================================
# 7. CALCULATE TREND STATISTICS
# ============================================================

def calculate_trend(data, column):

    """
    Linear regression:
    Temperature = intercept + slope * Year

    Slope converted to Celsius per decade.
    """

    x = data["Year"]
    y = data[column]

    regression = linregress(x, y)

    slope_per_decade = regression.slope * 10

    return {
        "Slope_C_per_decade": slope_per_decade,
        "Intercept": regression.intercept,
        "R_squared": regression.rvalue ** 2,
        "P_value": regression.pvalue,
        "Std_error": regression.stderr * 10
    }


# ============================================================
# 8. REGRESSION INTERACTION TEST
# ============================================================

def trend_difference_test(data):

    """
    Test whether warming trends differ between datasets.

    Model:

    Temperature ~ Year + Dataset + Year * Dataset

    The interaction term tests whether slopes differ.
    """

    long_data = pd.DataFrame({
        "Year": np.concatenate([
            data["Year"].values,
            data["Year"].values
        ]),

        "Temperature": np.concatenate([
            data["Rectangle_C"].values,
            data["Observation_C"].values
        ]),

        "Dataset": (
            ["Rectangle"] * len(data) +
            ["Observation"] * len(data)
        )
    })

    model = smf.ols(
        "Temperature ~ Year * Dataset",
        data=long_data
    ).fit()

    interaction_name = [
        term for term in model.params.index
        if "Year:Dataset" in term
    ]

    if len(interaction_name) > 0:

        interaction = interaction_name[0]

        slope_difference = model.params[interaction]

        p_interaction = model.pvalues[interaction]

    else:

        slope_difference = np.nan
        p_interaction = np.nan

    return {
        "Slope_difference_C_per_year":
            slope_difference,

        "Slope_difference_C_per_decade":
            slope_difference * 10,

        "Trend_difference_p":
            p_interaction,

        "Model_R_squared":
            model.rsquared
    }


# ============================================================
# 9. COMPARISON STATISTICS
# ============================================================

def calculate_comparison_statistics(data):

    rectangle = data["Rectangle_C"]
    observation = data["Observation_C"]

    difference = rectangle - observation

    correlation = linregress(
        rectangle,
        observation
    )

    stats = {

        "N": len(data),

        "Correlation_r":
            correlation.rvalue,

        "Correlation_p":
            correlation.pvalue,

        "Scatter_R_squared":
            correlation.rvalue ** 2,

        "Mean_bias_Rectangle_minus_Observation":
            difference.mean(),

        "MAE":
            mean_absolute_error(
                observation,
                rectangle
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    observation,
                    rectangle
                )
            ),

        "Mean_difference":
            difference.mean(),

        "Difference_std":
            difference.std()
    }

    return stats


# ============================================================
# 10. PLOT TIME SERIES
# ============================================================

def plot_timeseries(
    data,
    period_name,
    rectangle_trend,
    observation_trend
):

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        data["Year"],
        data["Rectangle_anomaly"],
        color="steelblue",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="NOAA rectangle"
    )

    ax.plot(
        data["Year"],
        data["Observation_anomaly"],
        color="darkorange",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Observation"
    )

    # Rectangle trend
    x = data["Year"]

    rect_fit = (
        rectangle_trend["Intercept"] +
        rectangle_trend["Slope_C_per_decade"] / 10
        * x
    )

    # Since anomaly intercept needs recalculation,
    # calculate directly using regression on anomaly
    rect_reg = linregress(
        data["Year"],
        data["Rectangle_anomaly"]
    )

    obs_reg = linregress(
        data["Year"],
        data["Observation_anomaly"]
    )

    ax.plot(
        x,
        rect_reg.intercept + rect_reg.slope * x,
        color="steelblue",
        linestyle="--",
        linewidth=2
    )

    ax.plot(
        x,
        obs_reg.intercept + obs_reg.slope * x,
        color="darkorange",
        linestyle="--",
        linewidth=2
    )

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1
    )

    ax.set_title(
        f"{period_name}: NOAA Rectangle vs Observation "
        f"(1950–1999)",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Water Year (October–September)" if period_name == "Water Year" else "Year")
    ax.set_ylabel("Temperature anomaly (°C)")

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    filename = (
        f"{period_name.replace('-', '_')}_"
        "timeseries.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. PLOT SCATTER
# ============================================================

def plot_scatter(
    data,
    period_name,
    stats
):

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        data["Rectangle_C"],
        data["Observation_C"],
        s=45,
        color="steelblue",
        alpha=0.65,
        edgecolors="none"
    )

    # Regression
    regression = linregress(
        data["Rectangle_C"],
        data["Observation_C"]
    )

    x_fit = np.linspace(
        data["Rectangle_C"].min(),
        data["Rectangle_C"].max(),
        100
    )

    y_fit = (
        regression.intercept +
        regression.slope * x_fit
    )

    ax.plot(
        x_fit,
        y_fit,
        color="black",
        linewidth=2,
        label="Linear regression"
    )

    # 1:1 line
    min_val = min(
        data["Rectangle_C"].min(),
        data["Observation_C"].min()
    )

    max_val = max(
        data["Rectangle_C"].max(),
        data["Observation_C"].max()
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="grey",
        linewidth=1.5,
        label="1:1 line"
    )

    stats_text = (
        f"N = {stats['N']}\n"
        f"r = {stats['Correlation_r']:.3f}\n"
        f"R² = {stats['Scatter_R_squared']:.3f}\n"
        f"p = {stats['Correlation_p']:.2e}\n"
        f"RMSE = {stats['RMSE']:.2f} °C"
    )

    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.85
        )
    )

    ax.set_title(
        f"{period_name}: Rectangle vs Observation",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "NOAA rectangle temperature (°C)"
    )

    ax.set_ylabel(
        "Observational temperature (°C)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    filename = (
        f"{period_name.replace('-', '_')}_"
        "scatter.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. PLOT TEMPERATURE DIFFERENCE
# ============================================================

def plot_difference(
    data,
    period_name
):

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.scatter(
        data["Year"],
        data["Difference"],
        color="steelblue",
        alpha=0.65,
        s=35,
        edgecolors="none"
    )

    ax.plot(
        data["Year"],
        data["Difference"],
        color="steelblue",
        alpha=0.5,
        linewidth=1
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    mean_bias = data["Difference"].mean()

    ax.axhline(
        mean_bias,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Mean bias = {mean_bias:.2f} °C"
    )

    ax.set_title(
        f"{period_name}: Temperature Difference "
        "(Rectangle - Observation)",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("Water Year (October–September)" if period_name == "Water Year" else "Year")
    ax.set_ylabel("Temperature difference (°C)")

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    filename = (
        f"{period_name.replace('-', '_')}_"
        "difference.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. MAIN ANALYSIS
# ============================================================

all_results = []

periods = [
    "May-October",
    "November-October",
    "Water Year"
]


for period_name in periods:

    print("\n" + "=" * 60)

    print(
        f"Processing: {period_name}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # NOAA DATA
    # --------------------------------------------------------

    if period_name in NOAA_FILES:

        noaa_file = (
            BASE_DIR / NOAA_FILES[period_name]
        )

        noaa_data = read_noaa_file(
            noaa_file
        )

        rectangle_data = calculate_noaa_period(
            noaa_data,
            period_name
        )



    # --------------------------------------------------------
    # OBSERVATION DATA
    # --------------------------------------------------------

    observation_data = calculate_observation_period(
        obs_mean,
        period_name
    )


    # --------------------------------------------------------
    # MATCH DATA
    # --------------------------------------------------------

    matched = pd.merge(
        rectangle_data,
        observation_data,
        on="Year",
        how="inner"
    )

    matched = matched[
        (matched["Year"] >= START_YEAR) &
        (matched["Year"] <= END_YEAR)
    ].copy()

    matched = matched.sort_values(
        "Year"
    )


    if len(matched) == 0:

        print(
            f"No matched data for {period_name}"
        )

        continue


    # --------------------------------------------------------
    # CALCULATE ANOMALIES
    # --------------------------------------------------------

    matched["Rectangle_anomaly"] = (
        matched["Rectangle_C"] -
        matched["Rectangle_C"].mean()
    )

    matched["Observation_anomaly"] = (
        matched["Observation_C"] -
        matched["Observation_C"].mean()
    )


    # --------------------------------------------------------
    # CALCULATE TREND
    # --------------------------------------------------------

    rectangle_trend = calculate_trend(
        matched,
        "Rectangle_C"
    )

    observation_trend = calculate_trend(
        matched,
        "Observation_C"
    )


    # --------------------------------------------------------
    # TREND DIFFERENCE TEST
    # --------------------------------------------------------

    trend_test = trend_difference_test(
        matched
    )


    # --------------------------------------------------------
    # COMPARISON STATISTICS
    # --------------------------------------------------------

    comparison_stats = calculate_comparison_statistics(
        matched
    )


    # --------------------------------------------------------
    # SAVE MATCHED DATA
    # --------------------------------------------------------

    matched["Difference"] = (
        matched["Rectangle_C"] -
        matched["Observation_C"]
    )

    matched_file = (
        OUTPUT_DIR /
        f"{period_name.replace('-', '_')}_matched_data.csv"
    )

    matched.to_csv(
        matched_file,
        index=False
    )


    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_timeseries(
        matched,
        period_name,
        rectangle_trend,
        observation_trend
    )

    plot_scatter(
        matched,
        period_name,
        comparison_stats
    )

    plot_difference(
        matched,
        period_name
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    result = {

        "Period": period_name,

        "N": len(matched),

        "Rectangle_trend_C_per_decade":
            rectangle_trend["Slope_C_per_decade"],

        "Rectangle_R_squared":
            rectangle_trend["R_squared"],

        "Rectangle_p":
            rectangle_trend["P_value"],

        "Observation_trend_C_per_decade":
            observation_trend["Slope_C_per_decade"],

        "Observation_R_squared":
            observation_trend["R_squared"],

        "Observation_p":
            observation_trend["P_value"],

        "Trend_difference_C_per_decade":
            trend_test[
                "Slope_difference_C_per_decade"
            ],

        "Trend_difference_p":
            trend_test[
                "Trend_difference_p"
            ],

        "Correlation_r":
            comparison_stats[
                "Correlation_r"
            ],

        "Correlation_p":
            comparison_stats[
                "Correlation_p"
            ],

        "Scatter_R_squared":
            comparison_stats[
                "Scatter_R_squared"
            ],

        "Mean_bias_C":
            comparison_stats[
                "Mean_bias_Rectangle_minus_Observation"
            ],

        "MAE_C":
            comparison_stats[
                "MAE"
            ],

        "RMSE_C":
            comparison_stats[
                "RMSE"
            ]
    }

    all_results.append(
        result
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nMatched years:", len(matched))

    print("\nRectangle trend:")
    print(rectangle_trend)

    print("\nObservation trend:")
    print(observation_trend)

    print("\nTrend difference:")
    print(trend_test)

    print("\nComparison statistics:")
    print(comparison_stats)


# ============================================================
# 14. SAVE SUMMARY TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_file = (
    OUTPUT_DIR /
    "NOAA_rectangle_observation_comparison_summary.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# 15. SAVE OBSERVATIONAL SPATIAL VARIABILITY
# ============================================================

obs_stdev.to_csv(
    OUTPUT_DIR /
    "Observational_spatial_variability_1950_1999.csv",
    index=False
)


print("\n" + "=" * 60)

print("ANALYSIS COMPLETE")

print("=" * 60)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSummary table:")
print(results_df)