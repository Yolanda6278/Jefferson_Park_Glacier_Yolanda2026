# ============================================================
# Oregon Glaciers: Regional NOAA Climate–AAR Regressions
# Five glaciers pooled together; glacier-specific colours.
#
# Climate datasets (NOAA regional data):
#   1. May-October mean temperature (°F)
#   2. November-October mean temperature (°F)
#   3. October-September mean temperature (°F)
#   4. November-April precipitation (inches)
#
# Temperature:
#   Convert °F to °C, then calculate anomalies relative to 1991-2020.
#
# Precipitation:
#   Convert inches to mm. Regression uses total seasonal precipitation.
#
# For each climate variable:
#   - all available glacier-year AAR observations are plotted
#   - each glacier is a different colour
#   - one pooled linear regression is fitted to all points
#   - N, R², p-value, slope, and intercept are reported
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

AAR_FILE = BASE_DIR / "all_AAR.csv"

NOAA_FILES = {
    "May_Oct_Temperature": (
        "May_October_6month_temp_ave_1895_2026(F).txt",
        "temperature",
        "May-October average temperature"
    ),
    "Nov_Oct_Temperature": (
        "northern_cascades_Nov_Oct_12month_temp_1895_2026(F).txt",
        "temperature",
        "November-October average temperature"
    ),
    "Oct_Sep_Temperature": (
        "Oct_Sep_12month_temp(F)_1895_2026.txt",
        "temperature",
        "October-September average temperature"
    ),
    "Nov_Apr_Precipitation": (
        "Nov_April_precipitation(in)_1895_2026.txt",
        "precipitation",
        "November-April precipitation"
    ),
}

OUTPUT_DIR = BASE_DIR / "output_NOAA_regional_regression"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_START = 1991
BASELINE_END = 2020

# ============================================================
# 2. READ AAR DATA
# ============================================================

aar_wide = pd.read_csv(AAR_FILE)
aar_wide.columns = aar_wide.columns.str.strip()

if "YEAR" not in aar_wide.columns:
    raise ValueError("AAR file must contain a 'YEAR' column.")

aar_wide["YEAR"] = pd.to_numeric(aar_wide["YEAR"], errors="coerce")

glacier_columns = [c for c in aar_wide.columns if c != "YEAR"]

for col in glacier_columns:
    aar_wide[col] = pd.to_numeric(aar_wide[col], errors="coerce")

aar_long = aar_wide.melt(
    id_vars="YEAR",
    value_vars=glacier_columns,
    var_name="Glacier",
    value_name="AAR"
)

aar_long = aar_long.dropna(subset=["YEAR", "AAR"]).copy()
aar_long["Year"] = aar_long["YEAR"].astype(int)

# Make glacier labels more readable
aar_long["Glacier"] = (
    aar_long["Glacier"]
    .str.replace("_AAR_interp", "", regex=False)
    .str.replace("_AAR", "", regex=False)
)

aar_long = aar_long[["Year", "Glacier", "AAR"]].sort_values(
    ["Year", "Glacier"]
)

print("\\n===== AAR DATA =====")
print(aar_long.head())
print("Glaciers:", aar_long["Glacier"].unique())
print("Total AAR observations:", len(aar_long))

# ============================================================
# 3. NOAA DATA READING FUNCTION
# ============================================================

def read_noaa_file(file_path, climate_type, climate_name):
    """
    Read NOAA text files with metadata lines beginning with '#'
    and data columns Date,Value.

    NOAA Date format is YYYYMM (e.g., 189510).
    The YYYY component is treated as the ending/label year of
    the climate period.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find NOAA file:\\n{file_path}\\n"
            "Check the filename and make sure it is in the same folder "
            "as this Python script."
        )

    df = pd.read_csv(file_path, comment="#")
    df.columns = df.columns.str.strip()

    if not {"Date", "Value"}.issubset(df.columns):
        raise ValueError(
            f"{file_path.name} must contain columns named Date and Value."
        )

    df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Date", "Value"]).copy()

    # Date is YYYYMM; use YYYY as the climate period label/year.
    df["Year"] = df["Date"].astype(int) // 100

    df = df[["Year", "Value"]].drop_duplicates("Year").sort_values("Year")

    if climate_type == "temperature":
        # NOAA temperature files are in Fahrenheit.
        df["Climate_C"] = (df["Value"] - 32) / 1.8

        baseline_mean = df.loc[
            df["Year"].between(BASELINE_START, BASELINE_END),
            "Climate_C"
        ].mean()

        df["Climate_Anomaly_C"] = df["Climate_C"] - baseline_mean

        df = df.rename(columns={
            "Value": "Climate_F",
            "Climate_C": "Climate_C"
        })

        predictor = "Climate_Anomaly_C"
        x_label = (
            f"{climate_name} anomaly (°C; "
            f"NOAA {BASELINE_START}-{BASELINE_END} baseline)"
        )
        units_description = "Temperature anomaly (°C)"

        print(f"\\n{climate_name}")
        print(f"1991-2020 baseline mean: {baseline_mean:.3f} °C")

    elif climate_type == "precipitation":
        # NOAA precipitation file is in inches.
        df["Climate_mm"] = df["Value"] * 25.4

        predictor = "Climate_mm"
        x_label = f"{climate_name} (mm)"
        units_description = "Precipitation (mm)"

        df = df.rename(columns={"Value": "Climate_inches"})

    else:
        raise ValueError("Unknown climate_type.")

    df["Climate_Name"] = climate_name

    return df, predictor, x_label, units_description

# ============================================================
# 4. REGRESSION AND PLOT FUNCTION
# ============================================================

def run_pooled_regression(
    matched,
    predictor,
    x_label,
    climate_key,
    climate_name,
    output_prefix
):
    valid = matched[["Year", "Glacier", "AAR", predictor]].dropna().copy()

    if len(valid) < 3:
        print(f"Skipping {climate_key}: fewer than 3 matched observations.")
        return None

    x = valid[predictor]
    y = valid["AAR"]

    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    p_value = result.pvalue
    slope_stderr = result.stderr

    print("\\n" + "=" * 60)
    print(f"POOLED REGRESSION: AAR vs {climate_name}")
    print("=" * 60)
    print(f"N (glacier-year observations): {len(valid)}")
    print(f"Number of glaciers: {valid['Glacier'].nunique()}")
    print(f"Number of years: {valid['Year'].nunique()}")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R²: {r_squared:.4f}")
    print(f"P-value: {p_value:.6g}")
    print(f"Standard error of slope: {slope_stderr:.6f}")

    # --------------------------------------------------------
    # Plot: glacier-specific colours + pooled regression line
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    glaciers = sorted(valid["Glacier"].unique())
    cmap = plt.get_cmap("tab10")

    for i, glacier in enumerate(glaciers):
        glacier_data = valid[valid["Glacier"] == glacier]

        ax.scatter(
            glacier_data[predictor],
            glacier_data["AAR"],
            s=45,
            alpha=0.65,
            color=cmap(i % 10),
            edgecolor=None,
            linewidth=0.35,
            label=glacier
        )

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    ax.plot(
        x_fit,
        y_fit,
        color="black",
        linewidth=2.5,
        label="Pooled linear regression"
    )

    stats_text = (
        f"N = {len(valid)}\n"
        f"R² = {r_squared:.3f}\n"
        f"p = {p_value:.3g}"
    )

    ax.text(
        0.03, 0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round, pad=0.15", facecolor="white", alpha=0.85)
    )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("AAR", fontsize=12)
    ax.set_title(
        f"Oregon Glaciers: AAR vs {climate_name}",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    plot_file = OUTPUT_DIR / f"{output_prefix}_scatter.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()

    # Export matched data for this climate variable
    matched_file = OUTPUT_DIR / f"{output_prefix}_matched_data.csv"
    valid.to_csv(matched_file, index=False)

    return {
        "Climate_Variable": climate_name,
        "Climate_Key": climate_key,
        "N_Glacier_Year_Observations": len(valid),
        "N_Glaciers": valid["Glacier"].nunique(),
        "N_Years": valid["Year"].nunique(),
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_squared,
        "p_value": p_value,
        "Slope_SE": slope_stderr,
        "First_Year": valid["Year"].min(),
        "Last_Year": valid["Year"].max(),
        "Plot": plot_file.name,
        "Matched_Data": matched_file.name
    }

# ============================================================
# 5. PROCESS ALL NOAA CLIMATE FILES AND RUN REGRESSIONS
# ============================================================

regression_results = []
climate_summaries = []

for climate_key, file_info in NOAA_FILES.items():

    filename, climate_type, climate_name = file_info
    climate_file = BASE_DIR / filename

    climate_df, predictor, x_label, units_description = read_noaa_file(
        climate_file,
        climate_type,
        climate_name
    )

    # Export processed climate series
    climate_summary_file = OUTPUT_DIR / f"{climate_key}_climate_summary.csv"
    climate_df.to_csv(climate_summary_file, index=False)

    climate_summaries.append(climate_df)

    # Match climate to every glacier-year AAR observation
    matched = aar_long.merge(
        climate_df,
        on="Year",
        how="inner"
    )

    matched = matched.dropna(subset=["AAR", predictor]).copy()
    matched = matched.sort_values(["Year", "Glacier"])

    # Run pooled regression and generate scatter plot
    result = run_pooled_regression(
        matched=matched,
        predictor=predictor,
        x_label=x_label,
        climate_key=climate_key,
        climate_name=climate_name,
        output_prefix=climate_key
    )

    if result is not None:
        regression_results.append(result)

# ============================================================
# 6. EXPORT SUMMARY RESULTS
# ============================================================

results_df = pd.DataFrame(regression_results)

results_file = OUTPUT_DIR / "NOAA_regional_regression_results.csv"
results_df.to_csv(results_file, index=False)

print("\\n===== SUMMARY OF ALL REGRESSIONS =====")
print(results_df.to_string(index=False))

# Export number of glacier observations available in each year
coverage = (
    aar_long.groupby("Year")["Glacier"]
    .nunique()
    .rename("Number_of_Glaciers_with_AAR")
    .reset_index()
)

coverage.to_csv(
    OUTPUT_DIR / "Annual_glacier_AAR_coverage.csv",
    index=False
)

print("\\nAnalysis complete!")
print(f"All output files saved in:\\n{OUTPUT_DIR}")

# ============================================================
# STATISTICAL NOTE
# ============================================================
# These are pooled ordinary least-squares regressions.
# Multiple glaciers in a given year share the same regional climate
# predictor, so glacier-year observations are not fully independent.
# These pooled plots/regressions are useful for visualising the
# regional relationship. For formal inference, consider a mixed-effects
# model or a year-clustered approach as a follow-up.
