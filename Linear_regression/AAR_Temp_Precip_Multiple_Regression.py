# ============================================================
# Mt. Jefferson Glacier
# Climate–AAR Multiple Linear Regression
#
# Models:
#   1. AAR ~ Calendar-year temperature anomaly + annual precipitation
#   2. AAR ~ Water-year temperature anomaly + annual precipitation
#
# Calendar year: January-December
# Water year: October-September, labelled by ending year
# Temperature predictors are anomalies relative to 1991-2020.
# Precipitation is retained as annual total (mm).
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AAR_FILE = BASE_DIR / "AAR_interpolation_extrapolation_16Nov.csv"
PRISM_FILE = BASE_DIR / (
    "PRISM_ppt_tmin_tmean_tmax_Preliminary_800m_197901_202610_44.6841_-121.8070.csv"
)

OUTPUT_DIR = BASE_DIR / "output_multiple_regression"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_START = 1991
BASELINE_END = 2020


# ============================================================
# 2. READ PRISM DATA
# ============================================================

climate = pd.read_csv(PRISM_FILE, skiprows=11)

climate.columns = ["Date", "Precipitation", "Tmin", "Tmean", "Tmax"]

climate["Date"] = pd.to_datetime(climate["Date"], format="%Y-%m")

for col in ["Precipitation", "Tmin", "Tmean", "Tmax"]:
    climate[col] = pd.to_numeric(climate[col], errors="coerce")

climate["Year"] = climate["Date"].dt.year
climate["Month"] = climate["Date"].dt.month
climate["Temperature"] = climate["Tmean"]

print("\\n===== PRISM CLIMATE DATA =====")
print(climate.head())
print("Climate period:", climate["Date"].min(), "to", climate["Date"].max())


# ============================================================
# 3. CALENDAR-YEAR CLIMATE (JANUARY-DECEMBER)
# ============================================================

annual_climate = climate.groupby("Year").agg(
    Annual_Temperature=("Temperature", "mean"),
    Annual_Precipitation=("Precipitation", "sum"),
    Annual_Temperature_Months=("Temperature", "count"),
    Annual_Precipitation_Months=("Precipitation", "count")
)

# Require 12 valid monthly observations for both variables
annual_climate.loc[
    annual_climate["Annual_Temperature_Months"] != 12,
    "Annual_Temperature"
] = np.nan

annual_climate.loc[
    annual_climate["Annual_Precipitation_Months"] != 12,
    "Annual_Precipitation"
] = np.nan


# ============================================================
# 4. WATER-YEAR CLIMATE (OCTOBER-SEPTEMBER)
# ============================================================

wy = climate.copy()
wy["Water_Year"] = wy["Year"]
wy.loc[wy["Month"] >= 10, "Water_Year"] += 1

water_year_climate = wy.groupby("Water_Year").agg(
    Water_Year_Temperature=("Temperature", "mean"),
    Water_Year_Precipitation=("Precipitation", "sum"),
    Water_Year_Temperature_Months=("Temperature", "count"),
    Water_Year_Precipitation_Months=("Precipitation", "count")
)

water_year_climate.loc[
    water_year_climate["Water_Year_Temperature_Months"] != 12,
    "Water_Year_Temperature"
] = np.nan

water_year_climate.loc[
    water_year_climate["Water_Year_Precipitation_Months"] != 12,
    "Water_Year_Precipitation"
] = np.nan

water_year_climate.index.name = "Year"


# ============================================================
# 5. TEMPERATURE ANOMALIES
# ============================================================

annual_baseline = annual_climate.loc[
    BASELINE_START:BASELINE_END, "Annual_Temperature"
].mean()

water_year_baseline = water_year_climate.loc[
    BASELINE_START:BASELINE_END, "Water_Year_Temperature"
].mean()

annual_climate["Annual_Temperature_Anomaly"] = (
    annual_climate["Annual_Temperature"] - annual_baseline
)

water_year_climate["Water_Year_Temperature_Anomaly"] = (
    water_year_climate["Water_Year_Temperature"] - water_year_baseline
)

print("\\n===== TEMPERATURE BASELINES =====")
print(f"Calendar-year baseline ({BASELINE_START}-{BASELINE_END}): {annual_baseline:.3f} °C")
print(f"Water-year baseline ({BASELINE_START}-{BASELINE_END}): {water_year_baseline:.3f} °C")


# ============================================================
# 6. COMBINE AND EXPORT CLIMATE DATA
# ============================================================

climate_summary = annual_climate.join(water_year_climate, how="outer").sort_index()
climate_summary.to_csv(OUTPUT_DIR / "Climate_summary.csv")

print("\\n===== CLIMATE SUMMARY =====")
print(climate_summary.head())


# ============================================================
# 7. READ AAR DATA
# ============================================================

aar = pd.read_csv(AAR_FILE)
aar.columns = aar.columns.str.strip()

aar["YEAR"] = pd.to_numeric(aar["YEAR"], errors="coerce")
aar["AAR"] = pd.to_numeric(aar["AAR"], errors="coerce")
aar = aar.dropna(subset=["YEAR", "AAR"])
aar["Year"] = aar["YEAR"].astype(int)
aar = (
    aar[["Year", "AAR"]]
    .sort_values("Year")
    .drop_duplicates("Year", keep="last")
)

print("\\n===== AAR DATA =====")
print(aar.head())
print("Missing AAR years:", sorted(set(range(1985, 2027)) - set(aar["Year"])))


# ============================================================
# 8. MATCH AAR AND CLIMATE
# ============================================================

merged = aar.merge(
    climate_summary,
    left_on="Year",
    right_index=True,
    how="left"
).sort_values("Year")

merged.to_csv(OUTPUT_DIR / "AAR_climate_matched.csv", index=False)

print("\\n===== MATCHED DATA =====")
print(merged.head())
print("Matched data shape:", merged.shape)


# ============================================================
# 9. MULTIPLE REGRESSION FUNCTION
# ============================================================

def perform_multiple_regression(
    data,
    predictors,
    model_name,
    predictor_labels=None,
    output_name=None
):
    """
    Fit OLS multiple linear regression:
        AAR = intercept + beta1*predictor1 + beta2*predictor2 + ...

    Outputs:
      - coefficient estimates, standard errors, p-values and 95% CIs
      - R-squared, adjusted R-squared, model F-test p-value, AIC and BIC
      - VIF for each predictor
      - observed-vs-predicted plot
      - residual plot
    """

    required_columns = ["AAR"] + predictors
    valid = data[required_columns].dropna().copy()

    if len(valid) <= len(predictors) + 1:
        print(f"Skipping {model_name}: insufficient complete observations.")
        return None, None, None

    X = valid[predictors]
    X = sm.add_constant(X, has_constant="add")
    y = valid["AAR"]

    model = sm.OLS(y, X).fit()

    print("\\n" + "=" * 65)
    print(f"MULTIPLE REGRESSION: {model_name}")
    print("=" * 65)
    print(model.summary())

    # VIF (excluding intercept)
    vif_rows = []
    X_vif = valid[predictors]
    for i, predictor in enumerate(predictors):
        vif_rows.append({
            "Model": model_name,
            "Predictor": predictor,
            "VIF": variance_inflation_factor(X_vif.values, i)
        })
    vif_df = pd.DataFrame(vif_rows)

    print("\\nVIF:")
    print(vif_df.to_string(index=False))

    # Coefficient table with confidence intervals
    conf_int = model.conf_int()
    coef_df = pd.DataFrame({
        "Model": model_name,
        "Term": model.params.index,
        "Coefficient": model.params.values,
        "Standard_Error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "CI_lower": conf_int.iloc[:, 0].values,
        "CI_upper": conf_int.iloc[:, 1].values
    })

    # Model-level statistics
    model_stats = {
        "Model": model_name,
        "N": int(model.nobs),
        "R_squared": model.rsquared,
        "Adjusted_R_squared": model.rsquared_adj,
        "F_statistic": model.fvalue,
        "Model_p_value": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic,
        "RMSE": np.sqrt(model.mse_resid)
    }

    # --------------------------------------------------------
    # Observed vs predicted plot
    # --------------------------------------------------------
    valid["Predicted_AAR"] = model.predict(X)
    valid["Residual"] = model.resid

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(valid["Predicted_AAR"], valid["AAR"], s=45, alpha=0.8)

    lower = min(valid["Predicted_AAR"].min(), valid["AAR"].min())
    upper = max(valid["Predicted_AAR"].max(), valid["AAR"].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black",
            linewidth=1.5, label="1:1 line")

    ax.set_xlabel("Predicted AAR", fontsize=12)
    ax.set_ylabel("Observed AAR", fontsize=12)
    ax.set_title(
        f"Observed vs Predicted AAR\\n{model_name}\\n"
        f"R² = {model.rsquared:.3f}; Adj. R² = {model.rsquared_adj:.3f}",
        fontsize=13
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_name:
        plt.savefig(
            OUTPUT_DIR / f"{output_name}_Observed_vs_Predicted.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

    # --------------------------------------------------------
    # Residual plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(valid["Predicted_AAR"], valid["Residual"], s=45, alpha=0.8)
    ax.axhline(0, linestyle="--", color="black", linewidth=1.2)
    ax.set_xlabel("Predicted AAR", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.set_title(f"Residuals: {model_name}", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_name:
        plt.savefig(
            OUTPUT_DIR / f"{output_name}_Residuals.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

    # Save the exact observations used in this model
    if output_name:
        valid.to_csv(OUTPUT_DIR / f"{output_name}_Model_Data.csv", index=False)

    return model_stats, coef_df, vif_df


# ============================================================
# 10. RUN MULTIPLE REGRESSION MODELS
# ============================================================

model_stats_list = []
coefficient_tables = []
vif_tables = []

# Model 1: Calendar-year temperature anomaly + calendar-year precipitation
stats, coefficients, vif = perform_multiple_regression(
    merged,
    predictors=[
        "Annual_Temperature_Anomaly",
        "Annual_Precipitation"
    ],
    model_name="Calendar-year Temperature + Annual Precipitation",
    output_name="Model1_Calendar_Temp_Annual_Precip"
)
if stats is not None:
    model_stats_list.append(stats)
    coefficient_tables.append(coefficients)
    vif_tables.append(vif)

# Model 2: Water-year temperature anomaly + water-year precipitation
stats, coefficients, vif = perform_multiple_regression(
    merged,
    predictors=[
        "Water_Year_Temperature_Anomaly",
        "Water_Year_Precipitation"
    ],
    model_name="Water-year Temperature + Water-year Precipitation",
    output_name="Model2_WaterYear_Temp_WaterYear_Precip"
)
if stats is not None:
    model_stats_list.append(stats)
    coefficient_tables.append(coefficients)
    vif_tables.append(vif)


# ============================================================
# 11. EXPORT RESULTS
# ============================================================

model_stats_df = pd.DataFrame(model_stats_list)
coefficient_df = (
    pd.concat(coefficient_tables, ignore_index=True)
    if coefficient_tables else pd.DataFrame()
)
vif_df = (
    pd.concat(vif_tables, ignore_index=True)
    if vif_tables else pd.DataFrame()
)

model_stats_df.to_csv(OUTPUT_DIR / "Multiple_Regression_Model_Statistics.csv", index=False)
coefficient_df.to_csv(OUTPUT_DIR / "Multiple_Regression_Coefficients.csv", index=False)
vif_df.to_csv(OUTPUT_DIR / "Multiple_Regression_VIF.csv", index=False)

print("\\n===== MODEL COMPARISON =====")
print(model_stats_df.to_string(index=False))

print("\\n===== COEFFICIENTS =====")
print(coefficient_df.to_string(index=False))

print("\\n===== VIF =====")
print(vif_df.to_string(index=False))

print("\\nAnalysis complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")
