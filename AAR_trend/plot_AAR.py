# ============================================================
# Mt. Jefferson Glacier Reconstruction
# Accumulation Area Ratio (AAR) Analysis
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "AAR_ver0.csv"

OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Select Year and AAR
df = df[["Year", "AAR"]].copy()

# Rename columns
df.columns = ["YEAR", "AAR"]

# Convert to numeric
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
df["AAR"] = pd.to_numeric(df["AAR"], errors="coerce")

# Remove invalid years
df = df.dropna(subset=["YEAR"])

df["YEAR"] = df["YEAR"].astype(int)

# Sort by year
df = df.sort_values("YEAR")

# Remove duplicate years
df = df.drop_duplicates(subset="YEAR", keep="last")


# ============================================================
# 3. Complete annual time series (1985-2026)
# ============================================================

all_years = pd.DataFrame({
    "YEAR": range(1985, 2027)
})

df = pd.merge(
    all_years,
    df,
    on="YEAR",
    how="left"
)

# AAR for 2012 remains NaN
# Missing years will appear as gaps in the plot


# ============================================================
# 4. Linear trend
# ============================================================

valid = df.dropna(subset=["AAR"]).copy()

slope, intercept = np.polyfit(
    valid["YEAR"],
    valid["AAR"],
    1
)

df["Trend"] = slope * df["YEAR"] + intercept

# R-squared
predicted = slope * valid["YEAR"] + intercept

ss_res = np.sum((valid["AAR"] - predicted) ** 2)

ss_tot = np.sum((valid["AAR"] - valid["AAR"].mean()) ** 2)

r_squared = 1 - ss_res / ss_tot


# ============================================================
# 5. Maximum and minimum AAR
# ============================================================

max_row = valid.loc[valid["AAR"].idxmax()]

min_row = valid.loc[valid["AAR"].idxmin()]


# ============================================================
# 6. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))


# ------------------------------------------------------------
# Background zones
# ------------------------------------------------------------

ax.axhspan(
    0, 0.3,
    color="red",
    alpha=0.20
)

ax.axhspan(
    0.6, 1.0,
    color="green",
    alpha=0.20
)


# Threshold lines
ax.axhline(
    0.3,
    color="red",
    linewidth=1,
    alpha=0.4
)

ax.axhline(
    0.6,
    color="green",
    linewidth=1,
    alpha=0.4
)


# ============================================================
# 7. AAR curve
# ============================================================

ax.plot(
    df["YEAR"],
    df["AAR"],
    color="black",
    marker="o",
    markersize=5,
    linewidth=1.8,
    label="AAR",
    zorder=3
)


# ============================================================
# 8. Missing data
# ============================================================

missing_years = df.loc[
    df["AAR"].isna(),
    "YEAR"
]

missing_label_added = False

for year in missing_years:

    previous = df.loc[
        df["YEAR"] == year - 1,
        "AAR"
    ].values

    following = df.loc[
        df["YEAR"] == year + 1,
        "AAR"
    ].values

    if (
        len(previous) > 0
        and len(following) > 0
        and not np.isnan(previous[0])
        and not np.isnan(following[0])
    ):

        ax.plot(
            [year - 1, year + 1],
            [previous[0], following[0]],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=(
                "Year with Missing Data"
                if not missing_label_added
                else None
            ),
            zorder=2
        )

        missing_label_added = True


# ============================================================
# 9. Linear trend
# ============================================================

ax.plot(
    df["YEAR"],
    df["Trend"],
    color="gray",
    linestyle="--",
    linewidth=2.5,
    label="Linear Trend"
)


# ============================================================
# 10. Maximum AAR
# ============================================================

ax.scatter(
    max_row["YEAR"],
    max_row["AAR"],
    color="green",
    s=100,
    zorder=5
)

ax.annotate(
    f'Max AAR ({int(max_row["YEAR"])}, '
    f'{max_row["AAR"]:.2f})',

    xy=(
        max_row["YEAR"],
        max_row["AAR"]
    ),

    xytext=(
        max_row["YEAR"] + 1,
        max_row["AAR"] + 0.10
    ),

    arrowprops=dict(
        arrowstyle="->",
        color="gray"
    ),

    fontsize=11,
    fontweight="bold"
)


# ============================================================
# 11. Minimum AAR
# ============================================================

ax.scatter(
    min_row["YEAR"],
    min_row["AAR"],
    color="red",
    s=100,
    zorder=5
)

ax.annotate(
    f'Min AAR ({int(min_row["YEAR"])}, '
    f'{min_row["AAR"]:.2f})',

    xy=(
        min_row["YEAR"],
        min_row["AAR"]
    ),

    xytext=(
        min_row["YEAR"] - 8,
        min_row["AAR"] + 0.10
    ),

    arrowprops=dict(
        arrowstyle="->",
        color="gray"
    ),

    fontsize=11,
    fontweight="bold"
)


# ============================================================
# 12. Background labels
# ============================================================

ax.text(
    1986, 0.95,
    "AAR > 0.6",
    color="darkgreen",
    fontsize=12,
    fontweight="bold"
)

ax.text(
    1986, 0.05,
    "AAR < 0.3",
    color="darkred",
    fontsize=12,
    fontweight="bold"
)


# ============================================================
# 13. Formatting
# ============================================================

ax.set_title(
    "Accumulation Area Ratio (AAR) Trend (1985-2026) for Jefferson Park Glacier",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Year",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "AAR Value",
    fontsize=12,
    fontweight="bold"
)

ax.set_xlim(1985, 2026)
ax.set_ylim(0, 1)

ax.set_xticks(np.arange(1985, 2027, 5))
ax.set_yticks(np.arange(0, 1.01, 0.2))

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend(
    loc="upper right",
    fontsize=11
)


# ============================================================
# 14. Save figures
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "AAR_trend_ver0.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.savefig(
#     OUTPUT_DIR / "AAR_over_time.pdf",
#     bbox_inches="tight"
# )

plt.show()


# ============================================================
# 15. Print statistics
# ============================================================

print("\n========== AAR Statistics ==========")

print(
    f"Maximum AAR: {max_row['AAR']:.3f} "
    f"({int(max_row['YEAR'])})"
)

print(
    f"Minimum AAR: {min_row['AAR']:.3f} "
    f"({int(min_row['YEAR'])})"
)

print(
    f"Linear trend slope: {slope:.5f} per year"
)

print(
    f"R-squared: {r_squared:.3f}"
)

print(
    f"Missing years: "
    f"{list(missing_years.astype(int))}"
)

print("\nFigures saved to:", OUTPUT_DIR)