import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AAR_FILE = BASE_DIR / "AAR_interpolation_extrapolation_19Nov.csv"

SNOTEL_FILE = BASE_DIR / "snotel_hogg_pass_mud_ridge.csv"

OUTPUT_DIR = BASE_DIR / "April_SWE_AAR_Ttest"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. READ AAR DATA
# ============================================================

aar_data = pd.read_csv(AAR_FILE)

aar_data = aar_data.rename(columns={
    "YEAR": "Year"
})

aar_data = aar_data[[
    "Year",
    "AAR"
]].copy()

aar_data["Year"] = pd.to_numeric(
    aar_data["Year"],
    errors="coerce"
)

aar_data["AAR"] = pd.to_numeric(
    aar_data["AAR"],
    errors="coerce"
)

aar_data = aar_data.dropna()


# ============================================================
# 3. READ SNOTEL APRIL SWE DATA
# ============================================================

snotel_data = pd.read_csv(SNOTEL_FILE)

snotel_data = snotel_data.rename(columns={
    snotel_data.columns[0]: "Year",
    snotel_data.columns[2]: "April_SWE_in"
})

snotel_data = snotel_data[[
    "Year",
    "April_SWE_in"
]].copy()

snotel_data["Year"] = pd.to_numeric(
    snotel_data["Year"],
    errors="coerce"
)

snotel_data["April_SWE_in"] = pd.to_numeric(
    snotel_data["April_SWE_in"],
    errors="coerce"
)

snotel_data = snotel_data.dropna()


# Convert inches to mm
snotel_data["April_SWE_mm"] = (
    snotel_data["April_SWE_in"] * 25.4
)


# ============================================================
# 4. MERGE DATA
# ============================================================

merged = pd.merge(
    aar_data,
    snotel_data,
    on="Year",
    how="inner"
)

merged = merged.sort_values("Year").reset_index(drop=True)


print("\n" + "="*60)
print("MERGED DATA")
print("="*60)

print(merged)

print(f"\nTotal matched years: {len(merged)}")


# ============================================================
# 5. FUNCTION: WELCH'S T-TEST
# ============================================================

def perform_ttest(data, high_threshold, low_threshold, analysis_name):

    # --------------------------------------------------------
    # Define groups
    # --------------------------------------------------------

    high_group = data[
        data["AAR"] > high_threshold
    ]

    low_group = data[
        data["AAR"] < low_threshold
    ]

    excluded_group = data[
        (data["AAR"] >= low_threshold) &
        (data["AAR"] <= high_threshold)
    ]


    high_swe = high_group["April_SWE_mm"].dropna()
    low_swe = low_group["April_SWE_mm"].dropna()


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    high_mean = high_swe.mean()
    low_mean = low_swe.mean()

    high_std = high_swe.std()
    low_std = low_swe.std()

    high_n = len(high_swe)
    low_n = len(low_swe)


    # --------------------------------------------------------
    # Welch's t-test
    # --------------------------------------------------------

    t_stat, p_value = ttest_ind(
        high_swe,
        low_swe,
        equal_var=False,
        nan_policy="omit"
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "="*60)
    print(analysis_name)
    print("="*60)

    print(f"\nHigh AAR group: AAR > {high_threshold}")
    print(f"n = {high_n}")
    print(f"Mean April SWE = {high_mean:.2f} mm")
    print(f"SD = {high_std:.2f} mm")

    print(f"\nLow AAR group: AAR < {low_threshold}")
    print(f"n = {low_n}")
    print(f"Mean April SWE = {low_mean:.2f} mm")
    print(f"SD = {low_std:.2f} mm")

    print(f"\nExcluded group: {low_threshold} <= AAR <= {high_threshold}")
    print(f"n = {len(excluded_group)}")

    print(f"\nWelch's t-test:")
    print(f"t-statistic = {t_stat:.3f}")
    print(f"P-value = {p_value:.4f}")


    # --------------------------------------------------------
    # Prepare boxplot data
    # --------------------------------------------------------

    boxplot_data = [
        low_swe,
        high_swe
    ]

    labels = [
        f"Low AAR\n(AAR < {low_threshold})\nn = {low_n}",
        f"High AAR\n(AAR > {high_threshold})\nn = {high_n}"
    ]


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    box = ax.boxplot(
        boxplot_data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6
        }
    )


    # --------------------------------------------------------
    # Box colors
    # --------------------------------------------------------

    colors = [
        "#4C78A8",  # Low AAR
        "#E07B39"   # High AAR
    ]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)


    # --------------------------------------------------------
    # Add individual data points
    # --------------------------------------------------------

    np.random.seed(42)

    for i, group in enumerate(boxplot_data, start=1):

        jitter = np.random.normal(
            0,
            0.04,
            size=len(group)
        )

        ax.scatter(
            np.full(len(group), i) + jitter,
            group,
            color="black",
            alpha=0.6,
            s=35,
            zorder=3
        )


    # --------------------------------------------------------
    # P-value annotation
    # --------------------------------------------------------

    ax.text(
        0.5,
        0.95,
        f"Welch's t-test\nP-value = {p_value:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="gray"
        )
    )


    # --------------------------------------------------------
    # Labels and title
    # --------------------------------------------------------

    ax.set_ylabel(
        "April Snow Water Equivalent (mm)",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_title(
        f"April SWE: High vs Low AAR\n"
        f"High threshold = {high_threshold}, "
        f"Low threshold = {low_threshold}",
        fontsize=16,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="-",
        alpha=0.3
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / (
        f"April_SWE_High_Low_AAR_"
        f"High_{high_threshold}_Low_{low_threshold}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nFigure saved to:\n{output_file}")


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "High_AAR_Threshold": high_threshold,
        "Low_AAR_Threshold": low_threshold,
        "High_AAR_n": high_n,
        "Low_AAR_n": low_n,
        "Excluded_n": len(excluded_group),
        "High_AAR_mean_SWE_mm": high_mean,
        "Low_AAR_mean_SWE_mm": low_mean,
        "High_AAR_SD_mm": high_std,
        "Low_AAR_SD_mm": low_std,
        "T_statistic": t_stat,
        "P_value": p_value
    }

# ============================================================
# 6. ANALYSIS 1
# HIGH AAR > 0.4
# LOW AAR < 0.35
# ============================================================

result_1 = perform_ttest(
    data=merged,
    high_threshold=0.40,
    low_threshold=0.35,
    analysis_name="HIGH AAR > 0.40 vs LOW AAR < 0.35"
)


# ============================================================
# 7. ANALYSIS 2
# HIGH AAR > 0.4
# LOW AAR < 0.3
# ============================================================

result_2 = perform_ttest(
    data=merged,
    high_threshold=0.40,
    low_threshold=0.30,
    analysis_name="HIGH AAR > 0.40 vs LOW AAR < 0.30"
)

# ============================================================
# 8. ANALYSIS 3
# HIGH AAR > 0.5
# LOW AAR < 0.3
# ============================================================

result_3 = perform_ttest(
    data=merged,
    high_threshold=0.50,
    low_threshold=0.30,
    analysis_name="HIGH AAR > 0.50 vs LOW AAR < 0.30"
)

# ============================================================
# 9. ANALYSIS 4
# HIGH AAR > 0.3
# LOW AAR < 0.3
# ============================================================

result_4 = perform_ttest(
    data=merged,
    high_threshold=0.30,
    low_threshold=0.30,
    analysis_name="HIGH AAR > 0.30 vs LOW AAR < 0.30"
)


# ============================================================
# 9. EXPORT RESULTS
# ============================================================

results_df = pd.DataFrame([
    result_1,
    result_2,
    result_3,
    result_4
])

results_file = OUTPUT_DIR / "April_SWE_AAR_Ttest_Results.csv"

results_df.to_csv(
    results_file,
    index=False
)


print("\n" + "="*60)
print("SUMMARY OF T-TEST RESULTS")
print("="*60)

print(results_df)

print(f"\nResults saved to:")
print(results_file)