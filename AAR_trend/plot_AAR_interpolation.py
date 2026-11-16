# ============================================================
# Mt. Jefferson Glacier Reconstruction
# AAR Analysis using Annual Accumulation Area + Glacier Area
# Glacier area: linear interpolation (1988-2023)
#              linear extrapolation (1985-1987, 2024-2026)
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ACCUMULATION_FILE = BASE_DIR / "accumulation_statistics_ver1.csv"
GLACIER_FILE = BASE_DIR / "glacier_area.csv"

OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

START_YEAR = 1985
END_YEAR = 2026
AAR_FILE = BASE_DIR / "AAR_interpolation_extrapolation_16Nov.csv"


# ============================================================
# 2. Load annual accumulation area
# ============================================================

acc = pd.read_csv(ACCUMULATION_FILE)

acc = acc[["YEAR", "SUM_Shape_Area"]].copy()
acc.columns = ["YEAR", "ACCUMULATION_AREA_M2"]

acc["YEAR"] = pd.to_numeric(acc["YEAR"], errors="coerce")
acc["ACCUMULATION_AREA_M2"] = pd.to_numeric(
    acc["ACCUMULATION_AREA_M2"], errors="coerce"
)

acc = acc.dropna(subset=["YEAR"])
acc["YEAR"] = acc["YEAR"].astype(int)
acc = acc.drop_duplicates(subset="YEAR", keep="last")

# Convert accumulation area from m² to km²
acc["ACCUMULATION_AREA_KM2"] = acc["ACCUMULATION_AREA_M2"] / 1_000_000


# ============================================================
# 3. Load mapped glacier area
# ============================================================

glacier = pd.read_csv(GLACIER_FILE)

glacier = glacier[["Year", "Shape_Area"]].copy()
glacier.columns = ["YEAR", "GLACIER_AREA_M2"]

glacier["YEAR"] = pd.to_numeric(glacier["YEAR"], errors="coerce")
glacier["GLACIER_AREA_M2"] = pd.to_numeric(
    glacier["GLACIER_AREA_M2"], errors="coerce"
)

glacier = glacier.dropna(subset=["YEAR", "GLACIER_AREA_M2"])
glacier["YEAR"] = glacier["YEAR"].astype(int)

# Convert glacier area from m² to km²
glacier["GLACIER_AREA_KM2"] = glacier["GLACIER_AREA_M2"] / 1_000_000

# Sort and remove duplicate mapped years
glacier = glacier.sort_values("YEAR")
glacier = glacier.drop_duplicates(subset="YEAR", keep="last")


# ============================================================
# 4. Interpolation and extrapolation of glacier area
# ============================================================

all_years = pd.DataFrame({"YEAR": np.arange(START_YEAR, END_YEAR + 1)})

# Merge mapped glacier area into the complete annual time series
df = all_years.merge(
    glacier[["YEAR", "GLACIER_AREA_KM2"]],
    on="YEAR",
    how="left"
)

# Interpolate missing years, and extrapolate beyond the first/last mapped years
df["GLACIER_AREA_KM2"] = (
    df["GLACIER_AREA_KM2"]
    .interpolate(method="linear", limit_area="inside")
)

# Linear extrapolation before the first mapped year
first_valid = glacier.iloc[0]
second_valid = glacier.iloc[1]

early_years = df["YEAR"] < int(first_valid["YEAR"])
early_slope = (
    second_valid["GLACIER_AREA_KM2"] - first_valid["GLACIER_AREA_KM2"]
) / (
    second_valid["YEAR"] - first_valid["YEAR"]
)

df.loc[early_years, "GLACIER_AREA_KM2"] = (
    first_valid["GLACIER_AREA_KM2"]
    + (df.loc[early_years, "YEAR"] - first_valid["YEAR"]) * early_slope
)

# Linear extrapolation after the last mapped year
last_valid = glacier.iloc[-1]
second_last_valid = glacier.iloc[-2]

late_years = df["YEAR"] > int(last_valid["YEAR"])
late_slope = (
    last_valid["GLACIER_AREA_KM2"] - second_last_valid["GLACIER_AREA_KM2"]
) / (
    last_valid["YEAR"] - second_last_valid["YEAR"]
)

df.loc[late_years, "GLACIER_AREA_KM2"] = (
    last_valid["GLACIER_AREA_KM2"]
    + (df.loc[late_years, "YEAR"] - last_valid["YEAR"]) * late_slope
)


# ============================================================
# 5. Merge accumulation area and calculate AAR
# ============================================================

df = df.merge(
    acc[["YEAR", "ACCUMULATION_AREA_KM2"]],
    on="YEAR",
    how="left"
)

# AAR = accumulation area / glacier area
df["AAR"] = (
    df["ACCUMULATION_AREA_KM2"] / df["GLACIER_AREA_KM2"]
)

# Keep AAR physically interpretable; do not silently modify values.
# Missing accumulation data (including 2012) remains NaN.
df.loc[df["ACCUMULATION_AREA_KM2"].isna(), "AAR"] = np.nan


# Flag years where glacier area is extrapolated
first_mapped_year = int(glacier["YEAR"].min())
last_mapped_year = int(glacier["YEAR"].max())

df["GLACIER_AREA_METHOD"] = np.where(
    df["YEAR"] < first_mapped_year,
    "Extrapolation",
    np.where(
        df["YEAR"] > last_mapped_year,
        "Extrapolation",
        np.where(
            df["YEAR"].isin(glacier["YEAR"]),
            "Mapped",
            "Interpolation"
        )
    )
)


# ============================================================
# 6. Linear trend and statistics (valid AAR years only)
# ============================================================

valid = df.dropna(subset=["AAR"]).copy()

if len(valid) < 2:
    raise ValueError("Not enough valid AAR values to calculate a linear trend.")

slope, intercept = np.polyfit(valid["YEAR"], valid["AAR"], 1)

df["Trend"] = slope * df["YEAR"] + intercept

predicted = slope * valid["YEAR"] + intercept
ss_res = np.sum((valid["AAR"] - predicted) ** 2)
ss_tot = np.sum((valid["AAR"] - valid["AAR"].mean()) ** 2)
r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

max_row = valid.loc[valid["AAR"].idxmax()]
min_row = valid.loc[valid["AAR"].idxmin()]


# ============================================================
# 7. Plot
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

# Background zones
ax.axhspan(0, 0.3, color="red", alpha=0.20)
ax.axhspan(0.6, 1.0, color="green", alpha=0.20)
ax.axhline(0.3, color="red", linewidth=1, alpha=0.4)
ax.axhline(0.6, color="green", linewidth=1, alpha=0.4)

# AAR curve
ax.plot(
    df["YEAR"], df["AAR"],
    color="black", marker="o", markersize=5, linewidth=1.8,
    label="AAR", zorder=3
)

# Missing AAR years: show a dashed bridge only for isolated missing years
missing_years = df.loc[df["AAR"].isna(), "YEAR"].tolist()
missing_label_added = False

for year in missing_years:
    previous = df.loc[df["YEAR"] == year - 1, "AAR"].values
    following = df.loc[df["YEAR"] == year + 1, "AAR"].values

    if (
        len(previous) > 0 and len(following) > 0
        and not np.isnan(previous[0]) and not np.isnan(following[0])
    ):
        ax.plot(
            [year - 1, year + 1],
            [previous[0], following[0]],
            color="red", linestyle="--", linewidth=1.5,
            label="Missing AAR data (2012)" if not missing_label_added else None,
            zorder=2
        )
        missing_label_added = True

# Linear trend
ax.plot(
    df["YEAR"], df["Trend"],
    color="gray", linestyle="--", linewidth=2.5,
    label="Linear Trend"
)

# Maximum and minimum
ax.scatter(max_row["YEAR"], max_row["AAR"], color="green", s=100, zorder=5)
ax.annotate(
    f'Max AAR ({int(max_row["YEAR"])}, {max_row["AAR"]:.2f})',
    xy=(max_row["YEAR"], max_row["AAR"]),
    xytext=(max_row["YEAR"] + 1, max_row["AAR"] + 0.10),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=11, fontweight="bold"
)

ax.scatter(min_row["YEAR"], min_row["AAR"], color="red", s=100, zorder=5)
ax.annotate(
    f'Min AAR ({int(min_row["YEAR"])}, {min_row["AAR"]:.2f})',
    xy=(min_row["YEAR"], min_row["AAR"]),
    xytext=(1990, 0.18),
    arrowprops=dict(
        arrowstyle="->",
        color="gray"
    ),
    fontsize=11,
    fontweight="bold",
    ha="right"
)

# Background labels
ax.text(1986, 0.95, "AAR > 0.6", color="darkgreen", fontsize=12, fontweight="bold")
ax.text(1986, 0.05, "AAR < 0.3", color="darkred", fontsize=12, fontweight="bold")

# Mark extrapolation periods
ax.axvspan(
    1985, first_mapped_year,
    color="lightgray",
    alpha=0.35,
    zorder=0
)

ax.axvspan(
    last_mapped_year, 2026,
    color="lightgray",
    alpha=0.35,
    zorder=0
)

ax.text(
    (1985 + first_mapped_year) / 2, 0.86,
    "Extrapolation", color="dimgray",
    fontsize=9, ha="center", va="center"
)
ax.text(
    (last_mapped_year + 2026) / 2, 0.86,
    "Extrapolation", color="dimgray",
    fontsize=9, ha="center", va="center"
)

# Formatting
ax.set_title(
    "Accumulation Area Ratio (AAR), 1985-2026\nJefferson Park Glacier",
    fontsize=15, fontweight="bold"
)
ax.set_xlabel("Year", fontsize=12, fontweight="bold")
ax.set_ylabel("AAR", fontsize=12, fontweight="bold")
ax.set_xlim(1985, 2026)
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(1985, 2027, 5))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "AAR_interpolation_extrapolation.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 8. Export annual data
# ============================================================

output_csv = OUTPUT_DIR / "AAR_interpolation_extrapolation.csv"
df.to_csv(output_csv, index=False)


# ============================================================
# 9. Print statistics
# ============================================================

print("\\n========== AAR Statistics ==========")
print(f"Maximum AAR: {max_row['AAR']:.3f} ({int(max_row['YEAR'])})")
print(f"Minimum AAR: {min_row['AAR']:.3f} ({int(min_row['YEAR'])})")
print(f"Linear trend slope: {slope:.5f} per year")
print(f"R-squared: {r_squared:.3f}")
print(f"Missing AAR years: {missing_years}")
print(f"Mapped glacier area years: {list(glacier['YEAR'].astype(int))}")
print(f"Interpolation period: {first_mapped_year}-{last_mapped_year}")
print(f"Extrapolation periods: {START_YEAR}-{first_mapped_year - 1} and {last_mapped_year + 1}-{END_YEAR}")
print("\\nFiles saved:")
print(f"Figure: {OUTPUT_DIR / 'AAR_interpolation_extrapolation.png'}")
print(f"CSV:    {output_csv}")
