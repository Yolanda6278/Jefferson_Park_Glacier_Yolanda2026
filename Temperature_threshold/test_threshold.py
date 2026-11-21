import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AAR_FILE = BASE_DIR / "AAR_interpolation_extrapolation_19Nov.csv"

SNOTEL_FILE = BASE_DIR / "snotel_hogg_pass_mud_ridge.csv"

SNOTEL_MAY_FILE = BASE_DIR / "snotel_hogg_pass_may1st.csv"

PRISM_FILE = BASE_DIR / "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv"

OUTPUT_DIR = BASE_DIR / "AAR_Climate_States_Statistics"
OUTPUT_DIR.mkdir(exist_ok=True)


# Baseline period
BASELINE_START = 1991
BASELINE_END = 2020


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

# Sort by Year (ascending), regardless of original file order
snotel_data = snotel_data.sort_values("Year").drop_duplicates(
    subset="Year",
    keep="last"
)

# Convert inches to mm
snotel_data["April_SWE_mm"] = (
    snotel_data["April_SWE_in"] * 25.4
)


# ============================================================
# 3.5 READ SNOTEL MAY 1 SWE DATA
# ============================================================

snotel_may_data = pd.read_csv(SNOTEL_MAY_FILE)

# May file format: Date, SWE (in); data are in ascending order
snotel_may_data["Date"] = pd.to_datetime(
    snotel_may_data.iloc[:, 0],
    errors="coerce"
)

snotel_may_data["Year"] = snotel_may_data["Date"].dt.year

snotel_may_data["May_1_SWE_in"] = pd.to_numeric(
    snotel_may_data.iloc[:, 1],
    errors="coerce"
)

snotel_may_data = snotel_may_data[[
    "Year",
    "May_1_SWE_in"
]].dropna()

snotel_may_data["Year"] = snotel_may_data["Year"].astype(int)

# Convert inches to mm
snotel_may_data["May_1_SWE_mm"] = (
    snotel_may_data["May_1_SWE_in"] * 25.4
)

# Sort by Year (ascending), regardless of original file order
snotel_may_data = snotel_may_data.sort_values("Year").drop_duplicates(
    subset="Year",
    keep="last"
)

print("\nHogg Pass May 1st SWE data:")
print(snotel_may_data.head())


# ============================================================
# 4. READ PRISM DATA
# ============================================================

climate = pd.read_csv(
    PRISM_FILE,
    skiprows=10
)

# Clean column names
climate.columns = climate.columns.str.strip()

# Rename columns for easier processing
climate = climate.rename(columns={
    "Date": "Date",
    "ppt (mm)": "Precipitation",
    "tmin (degrees C)": "Tmin",
    "tmean (degrees C)": "Tmean",
    "tmax (degrees C)": "Tmax"
})

# Convert Date to datetime
climate["Date"] = pd.to_datetime(
    climate["Date"],
    format="%Y-%m",
    errors="coerce"
)

# Convert climate variables to numeric
for column in [
    "Precipitation",
    "Tmin",
    "Tmean",
    "Tmax"
]:
    climate[column] = pd.to_numeric(
        climate[column],
        errors="coerce"
    )

# Remove invalid rows
climate = climate.dropna(
    subset=["Date", "Tmean"]
)

# Extract year and month
climate["Year"] = climate["Date"].dt.year
climate["Month"] = climate["Date"].dt.month

print("\nPRISM data loaded successfully.")

print(climate.head())

print(f"\nDate range: {climate['Date'].min()} to {climate['Date'].max()}")


# ============================================================
# 5. CALCULATE WATER YEAR AVERAGE TEMPERATURE
# ============================================================

# Water year:
# October of previous calendar year
# through September of current calendar year

climate["Water_Year"] = climate["Year"]

climate.loc[
    climate["Month"] >= 10,
    "Water_Year"
] += 1


water_year_climate = climate.groupby(
    "Water_Year"
).agg(
    WYA=("Tmean", "mean"),
    Temperature_Months=("Tmean", "count")
)


# Require all 12 months
water_year_climate.loc[
    water_year_climate["Temperature_Months"] != 12,
    "WYA"
] = np.nan


water_year_climate.index.name = "Year"


# ============================================================
# 6. CALCULATE WATER YEAR TEMPERATURE ANOMALY
# ============================================================

baseline = water_year_climate.loc[
    BASELINE_START:BASELINE_END,
    "WYA"
].mean()


water_year_climate["WYA_Anomaly"] = (
    water_year_climate["WYA"] - baseline
)


print("\n" + "="*60)
print("WATER YEAR TEMPERATURE")
print("="*60)

print(f"1991-2020 WYA baseline = {baseline:.3f} °C")


# ============================================================
# 7. MERGE DATASETS
# ============================================================

merged = aar_data.merge(
    water_year_climate,
    left_on="Year",
    right_index=True,
    how="inner"
)

merged = merged.merge(
    snotel_data,
    on="Year",
    how="left"
)

merged = merged.merge(
    snotel_may_data,
    on="Year",
    how="left"
)

merged = merged.sort_values(
    "Year"
).reset_index(drop=True)


print("\n" + "="*60)
print("MERGED DATA")
print("="*60)

print(merged)

print(f"\nMatched years: {len(merged)}")


# ============================================================
# 8. CLASSIFY AAR STATES
# ============================================================

def classify_aar(aar):

    if aar < 0.3:
        return "Low AAR (<0.3)"

    elif aar <= 0.6:
        return "Intermediate AAR (0.3-0.6)"

    else:
        return "High AAR (>0.6)"


merged["AAR_State"] = merged["AAR"].apply(
    classify_aar
)


# Set category order
state_order = [
    "Low AAR (<0.3)",
    "Intermediate AAR (0.3-0.6)",
    "High AAR (>0.6)"
]

merged["AAR_State"] = pd.Categorical(
    merged["AAR_State"],
    categories=state_order,
    ordered=True
)


print("\nAAR state counts:")

print(
    merged["AAR_State"].value_counts(
        sort=False
    )
)

# ============================================================
# 8.5 STATISTICAL TESTS
# ============================================================

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def perform_pairwise_tests(data, variable):

    comparisons = [
        ("Low AAR (<0.3)", "Intermediate AAR (0.3-0.6)"),
        ("Low AAR (<0.3)", "High AAR (>0.6)"),
        ("Intermediate AAR (0.3-0.6)", "High AAR (>0.6)")
    ]

    results = []

    raw_p_values = []

    for group_1, group_2 in comparisons:

        data_1 = data.loc[
            data["AAR_State"] == group_1,
            variable
        ].dropna()

        data_2 = data.loc[
            data["AAR_State"] == group_2,
            variable
        ].dropna()


        # Welch's t-test
        t_stat, p_value = ttest_ind(
            data_1,
            data_2,
            equal_var=False
        )


        results.append({
            "Variable": variable,
            "Comparison": f"{group_1} vs {group_2}",
            "Group_1_n": len(data_1),
            "Group_2_n": len(data_2),
            "t_statistic": t_stat,
            "Raw_p_value": p_value
        })

        raw_p_values.append(p_value)


    # Holm correction for multiple comparisons
    adjusted_p_values = multipletests(
        raw_p_values,
        method="holm"
    )[1]


    for i, adjusted_p in enumerate(adjusted_p_values):

        results[i]["Adjusted_p_value"] = adjusted_p


    return pd.DataFrame(results)


# ============================================================
# 8.6 P-VALUE FORMATTER
# ============================================================

def format_p_value(p):

    if p < 0.001:
        return "p < 0.001"

    else:
        return f"p = {p:.3f}"


# ============================================================
# 9. FUNCTION TO CALCULATE STATISTICS
# ============================================================

def calculate_statistics(data, variable, variable_name):

    results = []

    for state in state_order:

        group = data[
            data["AAR_State"] == state
        ][variable].dropna()

        n = len(group)

        mean = group.mean()

        sd = group.std()

        mean_plus_sd = mean + sd

        mean_minus_sd = mean - sd


        results.append({
            "AAR_State": state,
            "Variable": variable_name,
            "n": n,
            "Mean": mean,
            "SD": sd,
            "Mean_plus_SD": mean_plus_sd,
            "Mean_minus_SD": mean_minus_sd
        })


    return pd.DataFrame(results)


# ============================================================
# 10. CALCULATE WYA STATISTICS
# ============================================================

wya_stats = calculate_statistics(
    data=merged,
    variable="WYA",
    variable_name="Water Year Average Temperature"
)


wya_anomaly_stats = calculate_statistics(
    data=merged,
    variable="WYA_Anomaly",
    variable_name="Water Year Temperature Anomaly"
)


# ============================================================
# 11. CALCULATE APRIL SWE STATISTICS
# ============================================================

swe_stats = calculate_statistics(
    data=merged,
    variable="April_SWE_mm",
    variable_name="April SWE"
)

may_swe_stats = calculate_statistics(
    data=merged,
    variable="May_1_SWE_mm",
    variable_name="May 1st SWE"
)


# ============================================================
# 12. COMBINE STATISTICS
# ============================================================

statistics = pd.concat(
    [
        wya_stats,
        wya_anomaly_stats,
        swe_stats,
        may_swe_stats
    ],
    ignore_index=True
)


print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)

print(statistics)

# ============================================================
# 12.5 PAIRWISE P-VALUE ANALYSIS
# ============================================================

wya_tests = perform_pairwise_tests(
    merged,
    variable="WYA"
)

wya_anomaly_tests = perform_pairwise_tests(
    merged,
    variable="WYA_Anomaly"
)

swe_tests = perform_pairwise_tests(
    merged,
    variable="April_SWE_mm"
)

may_swe_tests = perform_pairwise_tests(
    merged,
    variable="May_1_SWE_mm"
)


pvalue_results = pd.concat(
    [
        wya_tests,
        wya_anomaly_tests,
        swe_tests,
        may_swe_tests
    ],
    ignore_index=True
)


print("\n" + "="*60)
print("PAIRWISE TEST RESULTS")
print("="*60)

print(pvalue_results)


# ============================================================
# 12.6 CREATE TEMPERATURE THRESHOLD TABLE FIGURE
# ============================================================

# Calculate the mean and standard deviation of WYA anomaly relative to
# the 1991-2020 baseline for the two AAR states used to define thresholds.
threshold_states = [
    "Low AAR (<0.3)",
    "Intermediate AAR (0.3-0.6)",
    "High AAR (>0.6)"
]

threshold_columns = [
    "AAR < 0.3",
    "0.3 <= AAR <= 0.6",
    "AAR > 0.6"
]

threshold_stats = []

for state in threshold_states:
    values = merged.loc[merged["AAR_State"] == state, "WYA_Anomaly"].dropna()
    mean = values.mean()
    sd = values.std()
    threshold_stats.append([mean, sd, mean + sd, mean - sd])

threshold_table_df = pd.DataFrame(
    {
        threshold_columns[0]: threshold_stats[0],
        threshold_columns[1]: threshold_stats[1],
        threshold_columns[2]: threshold_stats[2]
    },
    index=["Mean (x̄)", "Standard deviation (σ)", "x̄ + σ", "x̄ − σ"]
)

print("\nTemperature threshold table (WYA anomaly relative to 1991-2020):")
print(threshold_table_df)

# Create a publication-style table image
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.axis("off")

cell_text = [
    [f"{value:.6f}" if pd.notna(value) else "NA" for value in row]
    for row in threshold_table_df.values
]

table = ax.table(
    cellText=cell_text,
    rowLabels=threshold_table_df.index,
    colLabels=threshold_table_df.columns,
    cellLoc="center",
    rowLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.15, 1.8)

# Style table cells
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(1.2)
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#E6E6E6")
    elif col == -1:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#F2F2F2")

ax.set_title(
    "Water Year Temperature Anomaly Thresholds by AAR State\n(1991–2020 baseline)",
    fontsize=16,
    fontweight="bold",
    pad=20
)

plt.tight_layout()

threshold_table_file = OUTPUT_DIR / "WYA_Anomaly_Temperature_Threshold_Table.png"
plt.savefig(threshold_table_file, dpi=300, bbox_inches="tight")
plt.close(fig)

# Also export the numerical table
threshold_table_csv = OUTPUT_DIR / "WYA_Anomaly_Temperature_Threshold_Table.csv"
threshold_table_df.to_csv(threshold_table_csv)

print(f"\nTemperature anomaly threshold table image saved to:\n{threshold_table_file}")
print(f"Temperature anomaly threshold table CSV saved to:\n{threshold_table_csv}")


# ============================================================
# 13. EXPORT STATISTICS
# ============================================================

statistics_file = OUTPUT_DIR / "AAR_State_Climate_Statistics.csv"

statistics.to_csv(
    statistics_file,
    index=False
)

print(f"\nStatistics saved to:")
print(statistics_file)

# Export p-value results

pvalue_file = OUTPUT_DIR / "Pairwise_Ttest_Results.csv"

pvalue_results.to_csv(
    pvalue_file,
    index=False
)

print(f"\nP-value results saved to:")
print(pvalue_file)


# ============================================================
# 14. EXPORT MERGED DATA
# ============================================================

merged_file = OUTPUT_DIR / "Merged_AAR_WYA_April_May_1st_SWE.csv"

merged.to_csv(
    merged_file,
    index=False
)

print(f"\nMerged data saved to:")
print(merged_file)


# ============================================================
# 15. PLOT WYA
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


wya_box_data = [
    merged.loc[
        merged["AAR_State"] == state,
        "WYA"
    ].dropna()
    for state in state_order
]


wya_labels = [
    f"Low AAR\n(<0.3)",
    f"Intermediate\n(0.3-0.6)",
    f"High AAR\n(>0.6)"
]


box = ax.boxplot(
    wya_box_data,
    tick_labels=wya_labels,
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


colors = [
    "#4C78A8",
    "#A0A0A0",
    "#E07B39"
]


for patch, color in zip(
    box["boxes"],
    colors
):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)


# Individual points
np.random.seed(42)

for i, group in enumerate(
    wya_box_data,
    start=1
):

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


ax.set_ylabel(
    "Water Year Average Temperature (°C)",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "Water Year Average Temperature by AAR State",
    fontsize=16,
    fontweight="bold"
)

ax.grid(
    axis="y",
    linestyle="-",
    alpha=0.3
)

ax.tick_params(
    axis="both",
    labelsize=12
)

# Pairwise p-values
p_text = (
    "Welch's t-test (Holm-adjusted)\n"
    f"Low vs Intermediate: "
    f"{format_p_value(wya_tests.iloc[0]['Adjusted_p_value'])}\n"
    f"Low vs High: "
    f"{format_p_value(wya_tests.iloc[1]['Adjusted_p_value'])}\n"
    f"Intermediate vs High: "
    f"{format_p_value(wya_tests.iloc[2]['Adjusted_p_value'])}"
)


ax.text(
    0.98,
    0.98,
    p_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
        edgecolor="gray"
    )
)

plt.tight_layout()


wya_plot_file = OUTPUT_DIR / "WYA_by_AAR_State.png"

plt.savefig(
    wya_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15.5 PLOT WYA ANOMALY
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


wya_anomaly_box_data = [
    merged.loc[
        merged["AAR_State"] == state,
        "WYA_Anomaly"
    ].dropna()
    for state in state_order
]


box = ax.boxplot(
    wya_anomaly_box_data,
    tick_labels=wya_labels,
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


# Box colors
for patch, color in zip(
    box["boxes"],
    colors
):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)


# Individual points
np.random.seed(42)

for i, group in enumerate(
    wya_anomaly_box_data,
    start=1
):

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


# Zero anomaly reference line
ax.axhline(
    0,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8
)


ax.set_ylabel(
    "Water Year Temperature Anomaly (°C)",
    fontsize=14,
    fontweight="bold"
)


ax.set_title(
    "Water Year Temperature Anomaly by AAR State\n"
    "(1991–2020 baseline)",
    fontsize=16,
    fontweight="bold"
)


ax.grid(
    axis="y",
    linestyle="-",
    alpha=0.3
)


ax.tick_params(
    axis="both",
    labelsize=12
)

# Pairwise p-values
p_text = (
    "Welch's t-test (Holm-adjusted)\n"
    f"Low vs Intermediate: "
    f"{format_p_value(wya_anomaly_tests.iloc[0]['Adjusted_p_value'])}\n"
    f"Low vs High: "
    f"{format_p_value(wya_anomaly_tests.iloc[1]['Adjusted_p_value'])}\n"
    f"Intermediate vs High: "
    f"{format_p_value(wya_anomaly_tests.iloc[2]['Adjusted_p_value'])}"
)


ax.text(
    0.98,
    0.98,
    p_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
        edgecolor="gray"
    )
)

plt.tight_layout()


wya_anomaly_plot_file = OUTPUT_DIR / "WYA_Anomaly_by_AAR_State.png"


plt.savefig(
    wya_anomaly_plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# ============================================================
# 16. PLOT APRIL SWE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


swe_box_data = [
    merged.loc[
        merged["AAR_State"] == state,
        "April_SWE_mm"
    ].dropna()
    for state in state_order
]


box = ax.boxplot(
    swe_box_data,
    tick_labels=wya_labels,
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


for patch, color in zip(
    box["boxes"],
    colors
):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)


# Individual points
np.random.seed(42)

for i, group in enumerate(
    swe_box_data,
    start=1
):

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


ax.set_ylabel(
    "April Snow Water Equivalent (mm)",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "April Mean SWE by AAR State (Hogg Pass)",
    fontsize=16,
    fontweight="bold"
)

ax.grid(
    axis="y",
    linestyle="-",
    alpha=0.3
)

ax.tick_params(
    axis="both",
    labelsize=12
)

# Pairwise p-values
p_text = (
    "Welch's t-test (Holm-adjusted)\n"
    f"Low vs Intermediate: "
    f"{format_p_value(swe_tests.iloc[0]['Adjusted_p_value'])}\n"
    f"Low vs High: "
    f"{format_p_value(swe_tests.iloc[1]['Adjusted_p_value'])}\n"
    f"Intermediate vs High: "
    f"{format_p_value(swe_tests.iloc[2]['Adjusted_p_value'])}"
)


ax.text(
    0.02,
    0.98,
    p_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
        edgecolor="gray"
    )
)

plt.tight_layout()


swe_plot_file = OUTPUT_DIR / "April_SWE_by_AAR_State.png"

plt.savefig(
    swe_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16.5 PLOT MAY 1st SWE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


may_swe_box_data = [
    merged.loc[
        merged["AAR_State"] == state,
        "May_1_SWE_mm"
    ].dropna()
    for state in state_order
]


box = ax.boxplot(
    may_swe_box_data,
    tick_labels=wya_labels,
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


for patch, color in zip(
    box["boxes"],
    colors
):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)


# Individual points
np.random.seed(42)

for i, group in enumerate(
    may_swe_box_data,
    start=1
):

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


ax.set_ylabel(
    "May 1st Snow Water Equivalent (mm)",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "May 1st SWE by AAR State (Hogg Pass)",
    fontsize=16,
    fontweight="bold"
)

ax.grid(
    axis="y",
    linestyle="-",
    alpha=0.3
)

ax.tick_params(
    axis="both",
    labelsize=12
)

# Pairwise p-values
p_text = (
    "Welch's t-test (Holm-adjusted)\n"
    f"Low vs Intermediate: "
    f"{format_p_value(may_swe_tests.iloc[0]['Adjusted_p_value'])}\n"
    f"Low vs High: "
    f"{format_p_value(may_swe_tests.iloc[1]['Adjusted_p_value'])}\n"
    f"Intermediate vs High: "
    f"{format_p_value(may_swe_tests.iloc[2]['Adjusted_p_value'])}"
)


ax.text(
    0.02,
    0.98,
    p_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
        edgecolor="gray"
    )
)

plt.tight_layout()


may_swe_plot_file = OUTPUT_DIR / "May_1st_SWE_by_AAR_State_Hogg_Pass.png"

plt.savefig(
    may_swe_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

print(f"\nWYA figure:")
print(wya_plot_file)

print(f"\nApril SWE figure:")
print(swe_plot_file)

print(f"\nMay 1st SWE figure (Hogg Pass):")
print(may_swe_plot_file)

print(f"\nStatistics:")
print(statistics_file)