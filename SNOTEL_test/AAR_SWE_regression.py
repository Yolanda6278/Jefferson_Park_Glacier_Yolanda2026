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

SNOTEL_FILES = {
    "Hogg Pass April": BASE_DIR / "snotel_hogg_pass_mud_ridge.csv",
    "Hogg Pass May 1st": BASE_DIR / "snotel_hogg_pass_may1st.csv",
    "Santiam Junction April": BASE_DIR / "snotel_santiam_junction_april.csv",
    "Santiam Junction May 1st": BASE_DIR / "snotel_santiam_junction_may1st.csv",
}

OUTPUT_DIR = BASE_DIR / "4_SNOTEL_SWE_AAR_Regression"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. READ AAR DATA
# ============================================================
aar_data = pd.read_csv(AAR_FILE)
aar_data = aar_data.rename(columns={"YEAR": "Year"})
aar_data = aar_data[["Year", "AAR"]].copy()
aar_data["Year"] = pd.to_numeric(aar_data["Year"], errors="coerce")
aar_data["AAR"] = pd.to_numeric(aar_data["AAR"], errors="coerce")
aar_data = aar_data.dropna().sort_values("Year")

print("\nAAR data:")
print(aar_data.head())

# ============================================================
# 3. READ AND STANDARDISE THE FOUR SNOTEL DATASETS
# ============================================================
def read_snotel(file_path, dataset_name):
    """Read either the original 3-column Hogg Pass April file or a 2-column date/SWE file."""
    df = pd.read_csv(file_path)

    # Original Hogg Pass April format: Year, Date, SWE
    if df.shape[1] >= 3 and "Year" in str(df.columns[0]):
        year = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        swe = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    else:
        # New format: Date, SWE (e.g. 1980-05-01, 23.1)
        date = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        year = date.dt.year
        swe = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    out = pd.DataFrame({"Year": year, "SWE_in": swe})
    out = out.dropna()
    out["Year"] = out["Year"].astype(int)
    out["SWE_mm"] = out["SWE_in"] * 25.4
    out = out.sort_values("Year").drop_duplicates(subset="Year", keep="last")
    out = out.rename(columns={"SWE_in": f"{dataset_name}_SWE_in", "SWE_mm": f"{dataset_name}_SWE_mm"})
    return out

snotel_datasets = {}
for name, path in SNOTEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path.name}")
    snotel_datasets[name] = read_snotel(path, name.replace(" ", "_"))
    print(f"\n{name} data:")
    print(snotel_datasets[name].head())
    print(f"Years available: {snotel_datasets[name]['Year'].min()}–{snotel_datasets[name]['Year'].max()}")

# ============================================================
# 4. MERGE AAR WITH EACH SNOTEL DATASET
# ============================================================
merged_all = aar_data.copy()

for name, df in snotel_datasets.items():
    merged_all = pd.merge(merged_all, df, on="Year", how="left")

merged_all = merged_all.sort_values("Year")
merged_all.to_csv(OUTPUT_DIR / "AAR_all_4_SNOTEL_datasets_merged.csv", index=False)

print("\nMerged dataset (AAR plus all four SWE datasets):")
print(merged_all.head())

# ============================================================
# 5. REGRESSION FOR EACH DATASET
# ============================================================
regression_results = []
plot_info = []

plot_settings = [
    ("Hogg Pass April", "Hogg_Pass_April_SWE_mm", "April SWE (mm)"),
    ("Hogg Pass May 1st", "Hogg_Pass_May_1st_SWE_mm", "May 1st SWE (mm)"),
    ("Santiam Junction April", "Santiam_Junction_April_SWE_mm", "April SWE (mm)"),
    ("Santiam Junction May 1st", "Santiam_Junction_May_1st_SWE_mm", "May 1st SWE (mm)"),
]

# Column names are generated from dataset names above.
for display_name, x_col, x_label in plot_settings:
    data = merged_all[["Year", "AAR", x_col]].dropna().sort_values("Year")
    x = data[x_col]
    y = data["AAR"]

    if len(data) < 2 or x.nunique() < 2:
        print(f"\nSkipping regression for {display_name}: insufficient data or no variation.")
        continue

    reg = linregress(x, y)
    result = {
        "Dataset": display_name,
        "n": len(data),
        "Slope": reg.slope,
        "Intercept": reg.intercept,
        "R_squared": reg.rvalue**2,
        "P_value": reg.pvalue,
        "Standard_error_slope": reg.stderr,
    }
    regression_results.append(result)
    plot_info.append((display_name, x_label, data, reg))

    print("\n" + "=" * 65)
    print(f"{display_name.upper()} SWE vs AAR LINEAR REGRESSION")
    print("=" * 65)
    print(f"n: {len(data)}")
    print(f"Slope: {reg.slope:.5f}")
    print(f"Intercept: {reg.intercept:.5f}")
    print(f"R²: {reg.rvalue**2:.4f}")
    print(f"P-value: {reg.pvalue:.4f}")
    print(f"Standard error: {reg.stderr:.5f}")

    # Individual plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, color="steelblue", edgecolor="steelblue", alpha=0.7, s=80, label="AAR")
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = reg.slope * x_line + reg.intercept
    line, = ax.plot(x_line, y_line, color="darkorange", linewidth=3, label="Linear regression")
    stats_handle = ax.plot([], [], linestyle="", label=f"n = {len(data)}\nP-value = {reg.pvalue:.4f}\nR² = {reg.rvalue**2:.2f}")[0]
    ax.legend(handles=[ax.collections[0], line, stats_handle], loc="lower right", fontsize=12)
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel("AAR", fontsize=14, fontweight="bold")
    ax.set_title(f"AAR vs {display_name} SWE", fontsize=18, fontweight="bold")
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    safe_name = display_name.replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"AAR_vs_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Export regression summary
summary = pd.DataFrame(regression_results)
summary.to_csv(OUTPUT_DIR / "Regression_summary_4_datasets.csv", index=False)

# ============================================================
# 6. COMBINED 2 x 2 FIGURE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (display_name, x_label, data, reg) in zip(axes, plot_info):
    x_col = data.columns[2]
    x = data[x_col]
    y = data["AAR"]
    ax.scatter(x, y, color="steelblue", edgecolor="steelblue", alpha=0.7, s=65)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, reg.slope * x_line + reg.intercept, color="darkorange", linewidth=2.5)
    ax.text(
        0.04, 0.96,
        f"n = {len(data)}\nR² = {reg.rvalue**2:.2f}\np = {reg.pvalue:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none")
    )
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("AAR", fontsize=12, fontweight="bold")
    ax.set_title(display_name, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)

fig.suptitle("AAR vs SNOTEL SWE: Comparison of Four Datasets", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "AAR_vs_4_SNOTEL_datasets_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 7. EXPORT YEAR-MATCHED DATA FOR EACH REGRESSION
# ============================================================
for display_name, x_col, _ in plot_settings:
    matched = merged_all[["Year", "AAR", x_col]].dropna().sort_values("Year")
    matched.to_csv(OUTPUT_DIR / f"Matched_data_{display_name.replace(' ', '_')}.csv", index=False)

print("\nAll analyses complete.")
print(f"Outputs saved in: {OUTPUT_DIR}")
print("Files created:")
print(" - 4 individual regression plots")
print(" - 1 combined 2x2 comparison figure")
print(" - Regression_summary_4_datasets.csv")
print(" - AAR_all_4_SNOTEL_datasets_merged.csv")
print(" - 4 matched-data CSV files")
