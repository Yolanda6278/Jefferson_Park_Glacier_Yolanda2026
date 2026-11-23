from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Load data
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

csv_file = BASE_DIR / "all_AAR.csv"

df = pd.read_csv(csv_file)

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

df = df.sort_values("YEAR").reset_index(drop=True)


# ============================================================
# Define glaciers and colours
# ============================================================

glacier_columns = {
    "Hayden Glacier": "Hayden_AAR_interp",
    "Collier Glacier": "Collier_AAR_interp",
    "Eliot Glacier": "Eliot_AAR_interp",
    "Reid Glacier": "Reid_AAR_interp",
    "Jefferson Park Glacier": "Jefferson_AAR_interp"
}

colors = {
    "Hayden Glacier": "#0072B2",
    "Collier Glacier": "#E69F00",
    "Eliot Glacier": "#009E73",
    "Reid Glacier": "#CC79A7",
    "Jefferson Park Glacier": "#333333"
}


# ============================================================
# Output directory
# ============================================================

output_dir = BASE_DIR / "Regional_AAR_Comparison"

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Function to plot AAR series
#
# Solid lines:
#   Consecutive valid years
#
# Dashed lines:
#   Missing years
#
# Both use the glacier's own colour
#
# No markers
# ============================================================

def plot_AAR_series(ax, years, values, color, linewidth=2):

    years = np.asarray(years)
    values = np.asarray(values, dtype=float)

    # Identify valid data points
    valid_indices = np.where(~np.isnan(values))[0]

    # Connect consecutive valid points
    for i in range(len(valid_indices) - 1):

        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        year1 = years[idx1]
        year2 = years[idx2]

        value1 = values[idx1]
        value2 = values[idx2]

        # ----------------------------------------------------
        # Continuous data: solid line
        # ----------------------------------------------------

        if (
            idx2 - idx1 == 1
            and year2 - year1 == 1
        ):

            ax.plot(
                [year1, year2],
                [value1, value2],
                color=color,
                linewidth=linewidth,
                linestyle="-",
                zorder=3
            )

        # ----------------------------------------------------
        # Missing year(s): dashed line
        # Same colour as the glacier
        # ----------------------------------------------------

        else:

            ax.plot(
                [year1, year2],
                [value1, value2],
                color=color,
                linestyle="--",
                linewidth=linewidth,
                zorder=3
            )

# ============================================================
# Function to create regional AAR comparison figure
# ============================================================

def create_AAR_comparison(
    data,
    glacier_columns,
    glacier_colors,
    title,
    output_filename,
    linewidth=2
):

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(16, 8))


    # --------------------------------------------------------
    # AAR threshold zones
    # --------------------------------------------------------

    ax.axhspan(
        0.6, 1.0,
        color="green",
        alpha=0.12,
        zorder=0
    )

    ax.axhspan(
        0.0, 0.3,
        color="red",
        alpha=0.12,
        zorder=0
    )


    # --------------------------------------------------------
    # AAR threshold lines
    # --------------------------------------------------------

    ax.axhline(
        0.6,
        color="green",
        linewidth=1.2,
        alpha=0.6,
        zorder=1
    )

    ax.axhline(
        0.3,
        color="red",
        linewidth=1.2,
        alpha=0.6,
        zorder=1
    )


    # --------------------------------------------------------
    # Plot individual glacier AAR series
    # --------------------------------------------------------

    for glacier, column in glacier_columns.items():

        plot_AAR_series(
            ax=ax,
            years=data["YEAR"],
            values=data[column],
            color=glacier_colors[glacier],
            linewidth=linewidth
        )


    # --------------------------------------------------------
    # Threshold labels
    # --------------------------------------------------------

    ax.text(
        1986,
        0.92,
        "AAR > 0.6",
        color="darkgreen",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        1986,
        0.055,
        "AAR < 0.3",
        color="darkred",
        fontsize=12,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Year",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(
        "AAR",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylim(0, 1.0)

    ax.set_xlim(
        data["YEAR"].min(),
        data["YEAR"].max()
    )

    ax.set_xticks(
        np.arange(
            int(data["YEAR"].min()),
            int(data["YEAR"].max()) + 1,
            5
        )
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_handles = []

    # Glacier legend
    for glacier in glacier_columns:

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=glacier_colors[glacier],
                linewidth=linewidth,
                label=glacier
            )
        )

    # Missing year legend
    # legend_handles.append(
    #     Line2D(
    #         [0],
    #         [0],
    #         color="red",
    #         linestyle="--",
    #         linewidth=linewidth,
    #         label="Missing year(s)"
    #     )
    # )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=11,
        frameon=True,
        framealpha=0.95
    )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()


    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    plt.savefig(
        output_dir / f"{output_filename}.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        output_dir / f"{output_filename}.pdf",
        bbox_inches="tight"
    )


    # --------------------------------------------------------
    # Display figure
    # --------------------------------------------------------

    plt.show()

    plt.close(fig)


# ============================================================
# VERSION 1 — Original annual AAR
# ============================================================

create_AAR_comparison(
    data=df,
    glacier_columns=glacier_columns,
    glacier_colors=colors,
    title="Regional Comparison of Accumulation Area Ratio (AAR)",
    output_filename="Regional_AAR_Annual",
    linewidth=2
)


# ============================================================
# VERSION 2 — 7-year centred rolling mean
# ============================================================

# Calculate 7-year centred rolling mean

df_rolling = df.copy()

for glacier, column in glacier_columns.items():

    df_rolling[column] = (
        df[column]
        .rolling(
            window=7,
            center=True,
            min_periods=4
        )
        .mean()
    )


# Plot 7-year rolling mean

create_AAR_comparison(
    data=df_rolling,
    glacier_columns=glacier_columns,
    glacier_colors=colors,
    title="Regional Comparison of AAR (7-year Rolling Mean)",
    output_filename="Regional_AAR_7yr_Rolling_Mean",
    linewidth=2.5
)


# ============================================================
# VERSION 3 — Regional Mean AAR
# 7-year rolling mean + individual glaciers
# ============================================================

# ------------------------------------------------------------
# Calculate regional mean and standard deviation
# ------------------------------------------------------------

aar_columns = list(glacier_columns.values())

df_regional = df_rolling.copy()

# Regional mean
df_regional["Regional_Mean_AAR"] = (
    df_regional[aar_columns]
    .mean(axis=1, skipna=True)
)

# Regional standard deviation
df_regional["Regional_SD_AAR"] = (
    df_regional[aar_columns]
    .std(axis=1, skipna=True)
)

# Mean ± 1 SD
df_regional["Regional_Upper"] = (
    df_regional["Regional_Mean_AAR"]
    + df_regional["Regional_SD_AAR"]
)

df_regional["Regional_Lower"] = (
    df_regional["Regional_Mean_AAR"]
    - df_regional["Regional_SD_AAR"]
)


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 8))


# ------------------------------------------------------------
# AAR threshold zones
# ------------------------------------------------------------

ax.axhspan(
    0.6, 1.0,
    color="green",
    alpha=0.12,
    zorder=0
)

ax.axhspan(
    0.0, 0.3,
    color="red",
    alpha=0.12,
    zorder=0
)


# ------------------------------------------------------------
# AAR threshold lines
# ------------------------------------------------------------

ax.axhline(
    0.6,
    color="green",
    linewidth=1.2,
    alpha=0.6,
    zorder=1
)

ax.axhline(
    0.3,
    color="red",
    linewidth=1.2,
    alpha=0.6,
    zorder=1
)


# ------------------------------------------------------------
# Regional variability: Mean ± 1 SD
# ------------------------------------------------------------

ax.fill_between(
    df_regional["YEAR"],
    df_regional["Regional_Lower"],
    df_regional["Regional_Upper"],
    color="grey",
    alpha=0.25,
    label="Regional Mean ± 1 SD",
    zorder=2
)


# ------------------------------------------------------------
# Plot individual glacier 7-year rolling mean
# ------------------------------------------------------------

for glacier, column in glacier_columns.items():

    plot_AAR_series(
        ax=ax,
        years=df_rolling["YEAR"],
        values=df_rolling[column],
        color=colors[glacier],
        linewidth=2.0
    )


# ------------------------------------------------------------
# Plot regional mean
# ------------------------------------------------------------

ax.plot(
    df_regional["YEAR"],
    df_regional["Regional_Mean_AAR"],
    color="black",
    linestyle="--",
    linewidth=3,
    label="Regional Mean AAR",
    zorder=5
)


# ------------------------------------------------------------
# Threshold labels
# ------------------------------------------------------------

ax.text(
    1986,
    0.92,
    "AAR > 0.6",
    color="darkgreen",
    fontsize=12,
    fontweight="bold"
)

ax.text(
    1986,
    0.055,
    "AAR < 0.3",
    color="darkred",
    fontsize=12,
    fontweight="bold"
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_title(
    "Regional Comparison of AAR "
    "(7-year Rolling Mean)",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Year",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "AAR",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylim(0, 1.0)

ax.set_xlim(
    df_regional["YEAR"].min(),
    df_regional["YEAR"].max()
)

ax.set_xticks(
    np.arange(
        int(df_regional["YEAR"].min()),
        int(df_regional["YEAR"].max()) + 1,
        5
    )
)

ax.tick_params(
    axis="both",
    labelsize=12
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.25
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

legend_handles = []

# Individual glaciers
for glacier in glacier_columns:

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colors[glacier],
            linewidth=2.0,
            linestyle="-",
            label=glacier
        )
    )


# Regional mean
legend_handles.append(
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=3,
        linestyle="--",
        label="Regional Mean AAR"
    )
)


# Regional variability
legend_handles.append(
    Line2D(
        [0],
        [0],
        color="grey",
        linewidth=0,
        marker="s",
        markersize=10,
        alpha=0.4,
        label="Regional Mean ± 1 SD"
    )
)


# # Missing years
# legend_handles.append(
#     Line2D(
#         [0],
#         [0],
#         color="black",
#         linestyle="--",
#         linewidth=1.8,
#         label="Missing year(s)"
#     )
# )


ax.legend(
    handles=legend_handles,
    loc="upper right",
    fontsize=11,
    frameon=True,
    framealpha=0.95
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

plt.savefig(
    output_dir / "Regional_Mean_AAR_7yr.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    output_dir / "Regional_Mean_AAR_7yr.pdf",
    bbox_inches="tight"
)


# ------------------------------------------------------------
# Display figure
# ------------------------------------------------------------

plt.show()

plt.close(fig)