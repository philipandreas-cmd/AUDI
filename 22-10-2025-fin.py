# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:47:46 2025

@author: phili
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from zoneinfo import ZoneInfo

from energy_charts_api import EnergyPricesAPI
api = EnergyPricesAPI(cache = True)

#%% LOAD PRICE DATA FROM API
print("\n=== DATA LOADING ===")
# --- Period & TZ ---
start_date = "2020-10-04"
end_date   = datetime.now().date() #2024-10-11"  

# --- ENERGY PRICES (not location-specific) ---
#URL_DK1 = f"https://api.energy-charts.info/price?bzn=DK1&start={start_date}&end={end_date}"
#dk1 = pd.read_json(URL_DK1)

dk1 = api.get_prices(start_date, end_date)

dk1["date"] = pd.to_datetime(dk1["unix_seconds"], unit="s") + pd.Timedelta("02:00:00")
dk1 = dk1[["date", "price"]].rename(columns={"price": "dk1_price"})

# --- QC ---
print("\n--- PRICE DATA QC ---")
print(f"DK1 rows: {len(dk1):,}")
print("Minutes:", dk1['date'].dt.minute.unique().tolist())
print("Duplicates:", int(dk1['date'].duplicated().sum()))
print("NaNs DK1:", dk1.isna().sum().to_dict())

# --- Resample to exact hourly steps (fix 15/30/45 min drift) ---
dk1 = (dk1.sort_values("date")
       .set_index("date")
       .resample("h")
       .mean(numeric_only=True)
       .asfreq("h")
       .reset_index()
)
dk1["dk1_price"] = dk1["dk1_price"].interpolate(limit=4, limit_direction="both")
print("\n--- CLEANED PRICE DATA ---")
print("Minutes:", dk1['date'].dt.minute.unique().tolist())
#%% LOAD WIND DATA FROM API
    
LAT_ESB, LON_ESB = 55.05, 9.74
tz_enc = "Europe%2FCopenhagen"

URL_WEATHER = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LAT_ESB}&longitude={LON_ESB}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=wind_speed_10m"
    f"&timezone={tz_enc}"
)
wjson = requests.get(URL_WEATHER, timeout=30).json()
if "hourly" not in wjson:
    raise ValueError(f"Open-Meteo weather error: {wjson.get('reason', 'missing hourly')}")
weather_df = pd.DataFrame({
    "date": pd.to_datetime(wjson["hourly"]["time"]),
    "wind_speed_10m": wjson["hourly"]["wind_speed_10m"]
})     # why did we do this, what error did we find?? - why not just remove it, what happens if we do
weather_df = pd.DataFrame({
    "date": pd.to_datetime(wjson["hourly"]["time"]),
    "wind_speed_10m": wjson["hourly"]["wind_speed_10m"]
})
weather_df["date"] = weather_df["date"] + pd.Timedelta("02:00:00")
weather_df["date"] = weather_df["date"].dt.floor("h").dt.tz_localize(None)
print("\n--- QC: WIND SPEED DATA ---")
print(f"Rows: {len(weather_df):,} | Range: {weather_df['date'].min()} → {weather_df['date'].max()}")
print("Minutes:", weather_df["date"].dt.minute.unique().tolist())
#%% --- Temperature ---

DK1_SITES = [
    ("Kassoe",         55.0250, 9.2561),
    ("Vandel_Airbase", 55.7350, 9.1960),
    ("Thyboron",       56.7030, 8.2150),
    ("Esbjerg",        55.05,   9.74),
    ]
    
temp_frames = []
for name, lat, lon in DK1_SITES:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max"
        f"&timezone={tz_enc}"
    )
    js = requests.get(url, timeout=30).json()
    if "daily" not in js:
        raise ValueError(f"Open-Meteo temp error ({name}): {js.get('reason', 'missing daily')}") # again why this error??
    temp_frames.append(
        pd.DataFrame({
            "date": pd.to_datetime(js["daily"]["time"]),
            f"max_temp_{name}": js["daily"]["temperature_2m_max"]
        })
    )

# Merge all cities + compute mean
temp_daily = temp_frames[0]
for df in temp_frames[1:]:
    temp_daily = temp_daily.merge(df, on="date", how="outer")
temp_daily["max_temp"] = temp_daily.filter(like="max_temp_").mean(axis=1)

print("\n--- QC: TEMPERATURE DATA ---")
print(f"Rows: {len(temp_daily):,} | Range: {temp_daily['date'].min()} → {temp_daily['date'].max()}")
print("Columns:", list(temp_daily.columns))
print("\n--- CLEANED TEMPERATURE DATA ---")

# Expand daily → hourly and convert to CET
temp_hourly = (
    temp_daily.set_index("date")
    .reindex(pd.date_range(temp_daily["date"].min(), temp_daily["date"].max(), freq="h"))
    .ffill()
    .rename_axis("date")
    .reset_index()
)
temp_hourly["date"] = temp_hourly["date"] + pd.Timedelta("02:00:00")
temp_hourly["date"] = temp_hourly["date"].dt.tz_localize(None)
print(f"Rows: {len(temp_hourly):,} | Range: {temp_hourly['date'].min()} → {temp_hourly['date'].max()}")
#%% LOAD DATA FROM CSV (Excel, HourUTC -> DK local time, robust to DST)

watt = pd.read_csv(
    "ProductionConsumptionSettlementnew.csv",
    parse_dates=["date"], dayfirst=True, thousands=",", skipinitialspace=True
)

watt.columns = ["date", "gross_consumption_mwh"]
watt["date"] = (
    pd.to_datetime(watt["date"], utc=True)        # --- Convert timezone (UTC → Europe/Copenhagen) ---
    .dt.tz_convert("Europe/Copenhagen")
    .dt.floor("h", ambiguous="infer", nonexistent="shift_forward")
    .dt.tz_localize(None)
    + pd.Timedelta("02:00:00")                     
)
watt["gross_consumption_mwh"] = watt["gross_consumption_mwh"].ffill().bfill()

print("\n--- QC: WATT CONSUMPTION ---")
print(f"Rows: {len(watt):,} | Range: {watt['date'].min()} → {watt['date'].max()}")
print("Minutes unique:", watt["date"].dt.minute.unique().tolist())
print("Duplicates:", int(watt["date"].duplicated().sum()))
print("NaNs:", watt.isna().sum().to_dict())

#%% ALIGN AND MERGE ALL SOURCES
print("\n=== ALIGN & MERGE ALL DATASETS ===")

common_start = max(dk1["date"].min(), weather_df["date"].min(),
                   temp_hourly["date"].min(), watt["date"].min())
common_end = min(dk1["date"].max(), weather_df["date"].max(),
                 temp_hourly["date"].max(), watt["date"].max())

for name, df in zip(["dk1", "weather_df", "temp_hourly", "watt"],
                    [dk1, weather_df, temp_hourly, watt]):
    globals()[name] = df[(df["date"] >= common_start) & (df["date"] <= common_end)].copy()
    globals()[name]["date"] = globals()[name]["date"].dt.tz_localize(None)

# Merge into unified hourly dataset
data = (dk1.merge(weather_df, on="date", how="left")
            .merge(temp_hourly[["date", "max_temp"]], on="date", how="left")
            .merge(watt, on="date", how="left"))


print("\n--- FINAL MERGE QC ---")
print(f"Rows: {len(data):,} | Range: {data['date'].min()} → {data['date'].max()}")
print("NaN counts:\n", data.isna().sum().to_string())
print("\nColumns:", list(data.columns))
# Check for duplicates and minute consistency
print("\nDuplicate timestamps:", int(data["date"].duplicated().sum()))
print("Minutes unique:", np.unique(data["date"].dt.minute))
print("Seconds unique:", np.unique(data["date"].dt.second))
print("\nSample:\n", data.head(5))

raw_future_consumption_data = None

#%% HANDLE MISSING CONSUMPTION (Dynamic Pattern Fill)
"""
The consumption dataset (WATT) ends before the weather and price data.
Here we fill missing hourly consumption dynamically based on recent
patterns, instead of using a flat forward-fill.
"""

print("\n--- STEP 4: HANDLE MISSING CONSUMPTION (Dynamic Pattern Fill) ---")

# Count missing before
missing_before = data["gross_consumption_mwh"].isna().sum()
print(f"Missing before fill: {missing_before}")

if missing_before > 0:
    # Find where real data ends
    last_real_idx = data["gross_consumption_mwh"].last_valid_index()
    last_real_date = data.loc[last_real_idx, "date"]
    last_real_value = data.loc[last_real_idx, "gross_consumption_mwh"]
    print(f"Last real consumption: {last_real_date} = {last_real_value:.2f} MWh")

    # Try using last 7 days of data as pattern
    recent_window = data.loc[
        (data["date"] <= last_real_date) &
        (data["date"] >= last_real_date - pd.Timedelta(days=25))
    ].dropna(subset=["gross_consumption_mwh"])

    # If pattern too short, fall back to last 48 hours or even 1 day
    if len(recent_window) < 24:
        print("⚠️ Not enough data in last 7 days. Falling back to last 48 hours.")
        recent_window = data.loc[
            (data["date"] <= last_real_date) &
            (data["date"] >= last_real_date - pd.Timedelta(days=2))
        ].dropna(subset=["gross_consumption_mwh"])

    # Still empty? Use flat value baseline
    if recent_window.empty:
        print("⚠️ No valid recent data found — using constant continuation.")
        pattern = np.array([last_real_value])
    else:
        pattern = recent_window["gross_consumption_mwh"].to_numpy()

    # Find missing indices
    mask_missing = data["gross_consumption_mwh"].isna()
    past_data_hole_size = mask_missing.sum()
    for i in range(24*7):
        mask_missing[len(mask_missing) + i] = True
    num_missing = mask_missing.sum()
    print(f"Missing entries to fill: {num_missing}")

    if num_missing > 0:
        # Repeat pattern to match missing length
        repeats = int(np.ceil(num_missing / len(pattern)))
        extended_pattern = np.tile(pattern, repeats)[:num_missing]

        # Add mild noise and seasonal trend (−0.2% per day)
        hours_missing = np.arange(num_missing)
        trend_factor = 1 - (hours_missing / 24 / 5) * 0.002  # small daily decrease
        noise = np.random.normal(0, 0.02, num_missing)  # ±2%
        synthetic_values = extended_pattern * trend_factor * (1 + noise)

        # Assign synthetic values
        data.loc[mask_missing, "gross_consumption_mwh"] = synthetic_values[:past_data_hole_size]
        raw_future_consumption_data = synthetic_values[past_data_hole_size:]
        
# Add timestamp index to predicted consumption data
starting_date = data.iloc[-1].date + timedelta(hours=1)
date_range_future_consumption = pd.date_range(start=starting_date, end=data.iloc[-1].date + timedelta(days=7), freq="H")
future_consumption_data = pd.DataFrame(raw_future_consumption_data, columns=["consumption"], index=date_range_future_consumption)

# Visualize real vs dynamically generated continuation
plt.figure(figsize=(12,4))

# Split dataset into real vs synthetic parts
last_real_date = data["gross_consumption_mwh"].last_valid_index()
cutoff_date = pd.Timestamp("2025-10-02 08:00:00")

real_data = data[data["date"] <= cutoff_date]
synthetic_data = data[data["date"] > cutoff_date]

# Plot both sections in different colors
plt.plot(real_data["date"], real_data["gross_consumption_mwh"],
         color="steelblue", label="Real Consumption (MWh)", linewidth=1.5)
plt.plot(synthetic_data["date"], synthetic_data["gross_consumption_mwh"],
         color="orange", label="Predicted Continuation", linewidth=2)

# Labels and styling
plt.title("Dynamic Continuation of Gross Consumption (WATT)")
plt.xlabel("Date")
plt.ylabel("MWh")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Count missing after
missing_after = data["gross_consumption_mwh"].isna().sum()
print(f"Missing after fill: {missing_after}")

# Show preview
print("\nPreview of filled continuation:")
print(data.tail(8)[["date", "gross_consumption_mwh"]])

#%% FEATURE ENGINEERING

print("\n=== FEATURE ENGINEERING ===")

# --- Target variable (1-hour ahead forecast) ---
y = data["dk1_price"]    

X = pd.DataFrame()
X["date"] = data["date"]

# --- TIME FEATURES ---
X["hour"] = data["date"].dt.hour                          # Daily consumption rhythm
X["dayofweek"] = data["date"].dt.dayofweek                # Weekly pattern (Mon=0, Sun=6)
X["is_weekend"] = X["dayofweek"].isin([5, 6]).astype(int)  # Binary weekend flag

# --- WEATHER FEATURES ---
X["max_temp"] = pd.to_numeric(data["max_temp"], errors="coerce")         
X["wind_speed_10m"] = pd.to_numeric(data["wind_speed_10m"], errors="coerce")

# --- CONSUMPTION (WATT) ---
X["watt"] = pd.to_numeric(data["gross_consumption_mwh"], errors="coerce")

# --- LAG FEATURES (Price history) ---
X["lag_1d"] = y.shift(24)    # Same hour previous day
X["lag_7d"] = y.shift(7*24)    # Same hour previous week

# --- REMOVE first 24 hours (lags are NaN here) ---
X = X.iloc[7*24:].reset_index(drop=True)
y = y.iloc[7*24:].reset_index(drop=True)

# --- QUALITY CHECK ---
print("\n--- FEATURE ENGINEERING SUMMARY ---")
print(f"Rows in feature matrix: {len(X)}")
print(f"Columns: {list(X.columns)}")
print("\nNaN values per column:\n", X.isna().sum().to_string())
print("\nSample of engineered features:\n", X.head(5))

import seaborn as sns
corr = X[["hour","dayofweek","max_temp","wind_speed_10m","watt","lag_1d","lag_7d"]].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix (Before Modeling)")
plt.show()
#%% FEATURE EXPLORATION 

print("\n=== FEATURE EXPLORATION ===")

# --- 1. Hourly & weekly price patterns ---
plt.figure(figsize=(12,5))
sns.boxplot(x="hour", y=y, data=X.join(y), palette="Blues")
plt.title("Price Distribution by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("DK1 Price [DKK/MWh]")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(9,5))
sns.boxplot(x="dayofweek", y=y, data=X.join(y), palette="coolwarm")
plt.title("Price Distribution by Day of Week")
plt.xlabel("Day of Week (0=Mon, 6=Sun)")
plt.ylabel("DK1 Price [DKK/MWh]")
plt.grid(True, alpha=0.3)
plt.show()

# --- 2. Relationship between weather & price ---
fig, ax = plt.subplots(1, 2, figsize=(14,5))

sns.scatterplot(x="wind_speed_10m", y=y, data=X.join(y), ax=ax[0], alpha=0.4, color="steelblue")
ax[0].set_title("Price vs Wind Speed")
ax[0].set_xlabel("Wind Speed (m/s)")
ax[0].set_ylabel("Price [DKK/MWh]")
ax[0].grid(True, alpha=0.3)

sns.scatterplot(x="watt", y=y, data=X.join(y), ax=ax[1], alpha=0.4, color="darkorange")
ax[1].set_title("Price vs Electricity Consumption")
ax[1].set_xlabel("Gross Consumption [MWh]")
ax[1].set_ylabel("Price [DKK/MWh]")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- INSIGHT SUMMARY ---")
print("- Prices are higher during morning and evening peaks (hourly patterns visible).")
print("- Weekends show slightly lower median prices due to lower industrial demand.")
print("- Strong negative correlation between wind speed and price. (high wind supply is lowering prices)")
print("- High consumption aligns with higher prices, confirming demand-driven market behavior.")
#%% SPLIT DATA + TRAIN RANDOM FOREST MODEL 
"""
Step 1: Split data into time-based training and test sets.
Step 2: Train the baseline Random Forest Regressor on DK1 price.
Step 3: Inspect model behavior and feature influence.
"""
print("\n=== TRAIN / TEST SPLIT ===")

X.drop(columns=["date"], inplace=True)

cutoff = int(len(y) * 0.8)
X_train = X.iloc[:cutoff]
y_train = y.iloc[:cutoff]
X_test  = X.iloc[cutoff:]
y_test  = y.iloc[cutoff:]

# --- Align align X and Y with consistent numbers of samples
min_len = min(len(y_test), len(X_test))
X_test = X_test.iloc[:min_len]
y_test = y_test.iloc[:min_len]

# --- QC summary ---
print(f"Total rows: {len(X):,}")
print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
print(f"Train mean: {y_train.mean():.2f} | Test mean: {y_test.mean():.2f}")

print("\n--- TIME PERIODS ---")
print(f"Training period: {data['date'].iloc[24]} → {data['date'].iloc[cutoff + 24 - 1]}")
print(f"Testing  period: {data['date'].iloc[cutoff + 24]} → {data['date'].iloc[-1]}")

#%% BASELINE RANDOM FOREST TRAINING 
print("\n=== BASELINE RANDOM FOREST TRAINING ===")

rf_model = RandomForestRegressor(
    n_estimators=1500,    # Grid search = best tuned tree count
    max_depth=20,         # Grid search = best parameter, prevents overfitting while allowing nonlinearities
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)

# --- QC: Training summary ---
print("✅ Model training complete.")
print(f"Training samples: {len(X_train):,} | Testing samples: {len(X_test):,}")

# --- Feature importance ---
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(8)

print("\n--- TOP FEATURE IMPORTANCES ---")
print(top_features.to_string())

plt.figure(figsize=(8,4))
top_features.plot(kind="barh", color="steelblue")
plt.title("Top Feature Importances (Baseline RF)")
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#%% MODEL OPTIMIZATION – SPIKE-WEIGHTED TRAINING 
"""
Re-train the model with higher weights on price spikes and drops.
Goal: make the model more sensitive to extreme price volatility.
"""

print("\n=== SPIKE-WEIGHTED RANDOM FOREST TRAINING ===")

# --- Compute standardized price deviations ---
zscore = (y_train - y_train.mean()) / y_train.std()

# --- Create custom sample weights ---
# --- Create weights ---
# Weight logic:
#   Normal range (< 1.5 std):   1× weight
#   Moderate spike/drop (1.5–2.5 std): 3× weight
#   Extreme spike/drop (> 2.5 std):    5× weight
weights = np.ones_like(zscore)
weights[(zscore > 1.5) & (zscore <= 2.5)] = 3.0
weights[(zscore < -1.5) & (zscore >= -2.5)] = 3.0
weights[zscore > 2.5] = 5.0
weights[zscore < -2.5] = 5.0

# --- QC printout ---
num_high = np.sum(zscore > 1.5)
num_low  = np.sum(zscore < -1.5)
print(f"High spikes identified: {num_high} | Deep drops identified : {num_low}")
print(f"Weight range: {weights.min()} → {weights.max()} | Mean weight: {weights.mean():.2f}")

# --- Train weighted model ---
rf_model_weighted = RandomForestRegressor(
    n_estimators=1500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model_weighted.fit(X_train, y_train, sample_weight=weights)

print("\n✅ Weighted Random Forest model trained successfully.")
print("Model now emphasizes volatility zones (spikes & drops).")

# --- Compare feature importance shift ---
importances_weighted = pd.Series(rf_model_weighted.feature_importances_, index=X_train.columns)
plt.figure(figsize=(8,4))
pd.concat([
    top_features.rename("Baseline"),
    importances_weighted.sort_values(ascending=False).head(8).rename("Weighted")
], axis=1).plot(kind="barh")
plt.title("Feature Importance Comparison: Baseline vs Weighted RF")
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#%% EVALUATE MODEL PERFORMANCE
"""
Evaluation of Random Forest performance using MAE, RMSE, and R².
Includes visual diagnostics to identify bias, variance, and residual trends.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates

# --- Compute metrics ---
mae = mean_absolute_error(y_test, rf_predict)
mse = mean_squared_error(y_test, rf_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, rf_predict)

# --- Print results ---
print("\n--- MODEL PERFORMANCE ---")
print(f"Mean Absolute Error (MAE):   {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score:                     {r2:.3f}")

# --- Plot 1: Actual vs Predicted over time ---
plt.figure(figsize=(14,6))
plt.title("DK1 Electricity Price: Actual vs Predicted")
test_dates = data.loc[y_test.index, "date"]

plt.plot(test_dates, y_test.values, label="Actual DK1", linewidth=1.5, color="black")
plt.plot(test_dates, rf_predict, label="Predicted DK1", linewidth=1.5, color="orange", alpha=0.8)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price [DKK/MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Predicted vs Actual (scatter) ---
plt.figure(figsize=(7,7))
plt.scatter(y_test, rf_predict, alpha=0.6, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual DK1 Price [DKK/MWh]")
plt.ylabel("Predicted DK1 Price [DKK/MWh]")
plt.title(f"Predicted vs Actual DK1 Prices (R² = {r2:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: Residuals over time ---
residuals = y_test.values - rf_predict
plt.figure(figsize=(14,6))
plt.plot(test_dates, residuals, color="purple", linewidth=1)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Residuals over Time (Actual - Predicted)")
plt.xlabel("Date")
plt.ylabel("Error [DKK/MWh]")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Residual distribution ---
plt.figure(figsize=(7,5))
plt.hist(residuals, bins=40, color="teal", alpha=0.7, edgecolor="black")
plt.title("Distribution of Residuals")
plt.xlabel("Prediction Error [DKK/MWh]")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Spike performance diagnostics
spike_threshold = np.percentile(y_test, 95)
drop_threshold  = np.percentile(y_test, 5)

spike_mask = y_test > spike_threshold
drop_mask  = y_test < drop_threshold

def eval_segment(mask, label):
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test[mask], rf_predict[mask])
    r2  = r2_score(y_test[mask], rf_predict[mask])
    print(f"{label:<15} | MAE={mae:.2f} | R²={r2:.3f} | N={mask.sum()}")

print("\n--- Spike/Drop Segment Evaluation ---")
eval_segment(spike_mask, "High spikes")
eval_segment(drop_mask,  "Low drops")
eval_segment(~(spike_mask | drop_mask), "Normal range")

#%% LOAD FUTURE WEATHER & MARKET DATA
"""
Loads future hourly weather and market data for the next 7 days.
These replace static 'latest_*' features in the forecast section.
"""

from datetime import date
import requests

# --- Define forecast window ---
forecast_days = 7
start_date_future = datetime(2025, 10, 24)  # date.today()
end_date_future = start_date_future + timedelta(days=forecast_days)
tz_enc = "Europe%2FCopenhagen"

# --- WEATHER FORECAST (Esbjerg) ---
LAT_ESB, LON_ESB = 55.05, 9.74
URL_WEATHER_FUTURE = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT_ESB}&longitude={LON_ESB}"
    f"&hourly=temperature_2m,wind_speed_10m"
    f"&timezone={tz_enc}"
)

weather_future_json = requests.get(URL_WEATHER_FUTURE, timeout=30).json()
if "hourly" not in weather_future_json:
    raise ValueError("⚠️ Missing hourly data in future weather forecast.")

future_weather = pd.DataFrame({
    "date": pd.to_datetime(weather_future_json["hourly"]["time"]),
    "future_temp": weather_future_json["hourly"]["temperature_2m"],
    "future_wind_speed": weather_future_json["hourly"]["wind_speed_10m"],
})

# --- MERGE ALL FUTURE DATA ---
future_features = (
    future_weather
    .set_index("date")
    .sort_index()
)
future_features.index = future_features.index + pd.Timedelta("02:00:00")
future_features.index = future_features.index.tz_localize(None)
future_weather = future_weather.set_index("date")

print(f"\n✅ Loaded future features data: {len(future_features):,} rows")
print(f"Range: {future_features.index.min()} → {future_features.index.max()}")

#%% STEP-FORWARD FORECAST (Next 7 Days)
"""
Generates a 7-day hourly forecast for DK1 electricity prices
using the trained Random Forest model with all matching features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bp_start_date = data["date"].max()
bp_end_date   = bp_start_date + pd.Timedelta(days=7)
date_range    = pd.date_range(start=bp_start_date, end=bp_end_date, freq="h")
date_range = date_range.tz_localize(None) + pd.Timedelta("02:00:00")

future_df = pd.DataFrame(index=date_range)
future_df["hour"]       = future_df.index.hour
future_df["dayofweek"]  = future_df.index.dayofweek
future_df["is_weekend"] = future_df.dayofweek.isin([5,6]).astype(int)

last_known = y.iloc[-24*7:].tolist()
preds = []

feature_names = list(X_train.columns)
print("\nModel feature order:", feature_names)

print("Future consumption index sample:", future_consumption_data.index[:5])
print("Forecast loop index sample:", future_df.index[:5])
print("Equal?:", future_consumption_data.index.equals(future_df.index))
print(f"Align check: forecast starts {future_df.index.min()}, consumption starts {future_consumption_data.index.min()}")

# --- Safe accessor for future consumption ---
def safe_get_consumption(df, ts):
    """Return consumption value for timestamp ts (nearest match if missing)."""
    try:
        return df.loc[ts].consumption
    except KeyError:
        nearest_idx = df.index.get_indexer([ts], method="nearest")[0]
        return df.iloc[nearest_idx].consumption

def safe_get_future(df, ts, col):
    """Return nearest available forecast value for given timestamp."""
    try:
        return df.loc[ts, col]
    except KeyError:
        nearest_idx = df.index.get_indexer([ts], method="nearest")[0]
        return df.iloc[nearest_idx][col]

# --- Forecast loop ---
for current_time in future_df.index:
    lag_1d = last_known[-24]
    lag_7d = last_known[-24*7]

    row_data = {
        "hour": current_time.hour,
        "dayofweek": current_time.dayofweek,
        "is_weekend": int(current_time.dayofweek in [5,6]),
        "max_temp": safe_get_future(future_features, current_time, "future_temp"),
        "wind_speed_10m": safe_get_future(future_features, current_time, "future_wind_speed"),
        "watt": safe_get_consumption(future_consumption_data, current_time),
        "lag_1d": lag_1d,
        "lag_7d": lag_7d
    }

    X_future = pd.DataFrame([row_data])[feature_names]
    pred = rf_model.predict(X_future)[0]

    preds.append(pred)
    last_known.append(pred)

future_df["predicted_dk1_price"] = preds

print("\n✅ Step-forward forecast generated successfully!")
print(future_df.head())


# --- Bias Correction ---
print("\n--- APPLYING BIAS CORRECTION ---")
bias = (y_train - rf_model_weighted.predict(X_train)).mean()
print(f"Training bias mean: {bias:.2f}")

future_df["predicted_dk1_price_corrected"] = future_df["predicted_dk1_price"] + bias

# --- Plot bias-corrected forecast ---
plt.figure(figsize=(12,5))
plt.plot(data["date"], data["dk1_price"], label="Historical DK1 Price", color="steelblue")
plt.plot(future_df.index, future_df["predicted_dk1_price"], label="Raw Forecast", color="orange", linestyle="--")
plt.plot(future_df.index, future_df["predicted_dk1_price_corrected"], label="Bias-Corrected Forecast", color="green")
plt.xlabel("Date")
plt.ylabel("DKK/MWh")
plt.title("DK1 Electricity Price Forecast (Next 7 Days, Bias-Corrected)")
plt.legend()
plt.grid(True)
plt.show()

# --- Print correction summary ---
print(f"Before correction: Mean = {future_df['predicted_dk1_price'].mean():.2f}")
print(f"After correction:  Mean = {future_df['predicted_dk1_price_corrected'].mean():.2f}")


# --- Plot forecast ---
plt.figure(figsize=(12,5))
plt.plot(data["date"], data["dk1_price"], label="Historical DK1 Price", color="steelblue")
plt.plot(future_df.index, future_df["predicted_dk1_price"], label="Forecast (7 Days)", color="orange")
plt.xlabel("Date")
plt.ylabel("DKK/MWh")
plt.title("DK1 Electricity Price Forecast (Next 7 Days)")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- FORECAST FEATURE VARIATION ---")
cols_to_check = [
    "max_temp", "cloudiness", "wind_speed_10m", "watt"]
for c in cols_to_check:
    if c in future_features.columns:
        std = future_features[c].std()
        print(f"{c:20s} std = {std:.3f}")
        
print("\n--- FORECAST FEATURE VARIATION ---")
future_cols_to_check = [
    "future_temp", "future_cloud_cover", "future_wind_speed"]
for c in future_cols_to_check:
    if c in future_features.columns:
        std = future_features[c].std()
        mean = future_features[c].mean()
        print(f"{c:25s} mean={mean:.2f} | std={std:.2f}")

future_features.plot(subplots=True, figsize=(10,6), title="Future Feature Dynamics")
plt.tight_layout()
plt.show()


#%% PRICE SUMMARY OUTPUT (Visualization + Table)
"""
Visual summary of predicted DK1 prices for production planning.
Shows:
1) Average predicted price per weekday & hour
2) 15 cheapest production windows
3) Heatmap visualization
"""

import seaborn as sns
import matplotlib.pyplot as plt

# --- Detect forecast column ---
pred_col = "predicted_dk1_price" if "predicted_dk1_price" in future_df.columns else "prediction"

# --- Prepare grouped averages ---
future_df["weekday_name"] = future_df.index.day_name()
price_summary = (
    future_df.groupby(["weekday_name", "hour"])[pred_col]
    .mean()
    .reset_index()
)

# --- Sort weekdays Monday → Sunday ---
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
price_summary["weekday_name"] = pd.Categorical(price_summary["weekday_name"], categories=weekday_order, ordered=True)
price_summary = price_summary.sort_values(["weekday_name", "hour"])

# --- Print cheapest 15 production slots ---
cheapest = price_summary.sort_values(pred_col).head(15)
print("\n💡 15 Cheapest predicted production slots (weekday + hour):")
print(cheapest.to_string(index=False))

# --- Plot: Line plot of hourly pattern for each weekday ---
plt.figure(figsize=(10,6))
sns.lineplot(data=price_summary, x="hour", y=pred_col, hue="weekday_name", palette="coolwarm", linewidth=2)
plt.title("Predicted DK1 Prices per Hour by Weekday")
plt.xlabel("Hour of Day")
plt.ylabel("Predicted Price [DKK/MWh]")
plt.legend(title="Weekday")
plt.grid(True, alpha=0.3)
plt.show()

# --- Plot: Heatmap (weekday × hour) ---
pivot = price_summary.pivot(index="hour", columns="weekday_name", values=pred_col)
plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="coolwarm_r", annot=False)
plt.title("Predicted DK1 Price Heatmap (Weekday × Hour)")
plt.xlabel("Weekday")
plt.ylabel("Hour of Day")
plt.show()

# --- Summary statistics ---
print(f"\n🔹 Mean predicted price: {future_df[pred_col].mean():.2f} DKK/MWh")
print(f"🔹 Lowest hourly price:  {future_df[pred_col].min():.2f} DKK/MWh")
print(f"🔹 Highest hourly price: {future_df[pred_col].max():.2f} DKK/MWh")

#%% MODEL TEST: FORECAST vs ACTUAL (4–11 October 2025)
"""
Compares the model's 7-day forecast (from 4–11 October 2025)
against actual DK1 prices fetched from the Energy-Charts API.
Evaluates using MAE, RMSE, R², and residual mean.
"""

# --- 1. Fetch actual DK1 prices from API ---
actual_start = "2025-10-17"
actual_end   = "2025-10-24"

URL_ACTUAL_DK1 = f"https://api.energy-charts.info/price?bzn=DK1&start={actual_start}&end={actual_end}"
actual_json = requests.get(URL_ACTUAL_DK1, timeout=420).json()

actual_prices = pd.DataFrame({
    "date": pd.to_datetime(actual_json["unix_seconds"], unit="s") + pd.Timedelta("02:00:00"),
    "actual_price": actual_json["price"]
})
actual_prices["date"] = actual_prices["date"].dt.floor("h").dt.tz_localize(None)
actual_prices = actual_prices.set_index("date")

# --- 2. Align forecast and actuals by timestamp ---
forecast_eval = future_df.copy()
forecast_eval.index = forecast_eval.index.floor("h")
forecast_eval = forecast_eval.rename(columns={"predicted_dk1_price": "forecast_price"})

comparison = (
    forecast_eval[["forecast_price"]]
    .merge(actual_prices, left_index=True, right_index=True, how="inner")
)

# --- 3. Compute evaluation metrics ---
mae_eval = mean_absolute_error(comparison["actual_price"], comparison["forecast_price"])
rmse_eval = np.sqrt(mean_squared_error(comparison["actual_price"], comparison["forecast_price"]))
r2_eval = r2_score(comparison["actual_price"], comparison["forecast_price"])
residual_mean = np.mean(comparison["actual_price"] - comparison["forecast_price"])

# --- 4. Print evaluation summary ---
print("\n--- FORECAST vs ACTUAL (4–11 October 2025) ---")
print(f"MAE:           {mae_eval:.2f}")
print(f"RMSE:          {rmse_eval:.2f}")
print(f"R²:            {r2_eval:.3f}")
print(f"Residual mean: {residual_mean:.2f}")

# --- 5. Plot Actual vs Forecast ---
plt.figure(figsize=(12,6))
plt.plot(comparison.index, comparison["actual_price"], label="Actual DK1 Price", color="black", linewidth=1.5)
plt.plot(comparison.index, comparison["forecast_price"], label="Forecasted DK1 Price", color="orange", linewidth=1.5, alpha=0.8)
plt.title("DK1 Electricity Price Forecast vs Actual (4–11 Oct 2025)")
plt.xlabel("Date")
plt.ylabel("DKK/MWh")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6. Plot Residuals over Time ---
residuals_eval = comparison["actual_price"] - comparison["forecast_price"]
plt.figure(figsize=(12,5))
plt.plot(comparison.index, residuals_eval, color="purple", linewidth=1)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Residuals Over Time (Actual - Forecast)")
plt.xlabel("Date")
plt.ylabel("Error [DKK/MWh]")
plt.grid(True)
plt.tight_layout()
plt.show()


#%% EXPORT WEEKDAY FORECAST (Next 7 Days)
"""
Exports only weekday (Monday–Friday) forecast hours for production planning.
Saves to 'next_week_prediction_weekdays.csv'
"""

# --- Detect prediction column ---
pred_col = "predicted_dk1_price" if "predicted_dk1_price" in future_df.columns else "prediction"

# --- Filter to weekdays (0 = Monday ... 4 = Friday) ---
weekday_df = future_df[future_df.index.dayofweek < 5].copy()

# --- Prepare export DataFrame ---
output_df = weekday_df.reset_index().rename(columns={
    "index": "datetime",
    pred_col: "forecasted_price_dkk_mwh"
})

# --- Save to CSV ---
#output_df.to_csv("test_11-18_model_noDECL_7dLAG_fin.csv", index=False)

# --- Preview ---
print("\n📦 Saved weekday-only forecast to 'next_week_prediction_weekdays.csv'")
print(output_df.head(20))  # show first 20 hours
print(f"\nGenerated {len(output_df)} hourly weekday predictions from {output_df['datetime'].min()} to {output_df['datetime'].max()}.")
