# Lab 3: Time Series Forecasting
# Student: Andrii Volynets
# Assigned dataset: EuStockMarkets.csv, column DAX
# This script is a plain-text .py version of the accompanying notebook.

# ==============================================================================
# Lab 3: Time Series Forecasting
# Student: Andrii Volynets
# Assigned dataset: EuStockMarkets.csv, column DAX
# Goal: one-step-ahead forecasting of the German DAX stock index using a leakage-safe Conv1D + LSTM workflow.
# ==============================================================================

# ==============================================================================
# 1. Imports and reproducibility
# The notebook follows the lab manual: exploratory analysis, chronological split, windowed dataset, Conv1D + LSTM model, metrics, and diagnostic plots.
# ==============================================================================

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

SEED = 51
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.rcParams["figure.figsize"] = (10, 5)

def plot_series(time, series, fmt="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], fmt, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

# ==============================================================================
# 2. Read the assigned dataset
# The task assignment table assigns EuStockMarkets.csv (column: DAX) to Andrii Volynets.
# ==============================================================================

DATA_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/EuStockMarkets.csv"

from pathlib import Path
local_csv = (Path(__file__).resolve().parent / "data") / "EuStockMarkets.csv"
df = pd.read_csv(local_csv if local_csv.is_file() else DATA_URL)
print(df.head())
print(df.shape)
print(df.columns)

series = df["DAX"].astype("float32").to_numpy()
time = np.arange(len(series), dtype=np.int32)

plt.figure(figsize=(12, 5))
plot_series(time, series, label="DAX")
plt.title("EuStockMarkets: DAX closing price")
plt.legend()
plt.show()

# ==============================================================================
# 3. Basic statistical analysis
# Rolling mean and rolling standard deviation help identify non-stationarity. Stock indices usually have strong trend and volatility clustering rather than fixed calendar seasonality.
# ==============================================================================

myseries = pd.Series(series, index=time, name="DAX")

rolling_mean = myseries.rolling(window=30).mean()
rolling_std = myseries.rolling(window=30).std()

plt.figure(figsize=(12, 5))
plt.plot(myseries, label="Original")
plt.plot(rolling_mean, label="Rolling mean (30 observations)")
plt.plot(rolling_std, label="Rolling std (30 observations)")
plt.title("Rolling statistics")
plt.xlabel("Observation")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plot_acf(series, lags=260, alpha=0.05, zero=False)
plt.ylim(-0.2, 1.05)
plt.title("Autocorrelation Function up to 260 lags")
plt.grid(True)
plt.show()

# ==============================================================================
# 4. Lag plots and seasonal decomposition
# Two lag plots are used: lag 1 for short-term persistence and lag 260 as an approximate trading-year lag.
# The series is not strongly seasonal/cyclical in the classical sense; it is dominated by upward trend, persistence, and market volatility.
# ==============================================================================

def lag_scatter(series, lag):
    values = pd.DataFrame({"t": series[:-lag], f"t+{lag}": series[lag:]})
    plt.figure(figsize=(6, 6))
    plt.scatter(values["t"], values[f"t+{lag}"], alpha=0.5, s=12)
    plt.xlabel("t")
    plt.ylabel(f"t+{lag}")
    plt.title(f"Lagged scatter plot: t vs t+{lag}")
    plt.grid(True)
    plt.show()

lag_scatter(series, lag=1)
lag_scatter(series, lag=260)

decomp = seasonal_decompose(series, model="additive", period=260, extrapolate_trend="freq")
fig = decomp.plot()
fig.set_size_inches((14, 8))
fig.suptitle("Seasonal decomposition: additive model, period=260", y=1.02)
plt.show()

# ==============================================================================
# 5. Leakage-safe train/validation split
# The split preserves chronological order. The first 80% is used for training and the final 20% for validation.
# All normalization parameters are computed only from the training segment.
# ==============================================================================

split_time = int(len(series) * 0.80)

time_train = time[:split_time]
x_train_raw = series[:split_time]
time_valid = time[split_time:]
x_valid_raw = series[split_time:]

train_mean = x_train_raw.mean()
train_std = x_train_raw.std()

series_norm = (series - train_mean) / train_std
x_train = series_norm[:split_time]
x_valid = series_norm[split_time:]

plt.figure(figsize=(12, 5))
plot_series(time_train, x_train_raw, label="Train")
plot_series(time_valid, x_valid_raw, label="Validation")
plt.axvline(split_time, linestyle="--", label="Split")
plt.title("Train/validation split")
plt.legend()
plt.show()

print("split_time:", split_time)
print("train length:", len(x_train_raw), "validation length:", len(x_valid_raw))
print("train mean/std:", train_mean, train_std)

# ==============================================================================
# 6. Major functions from the script
#   - windowed_dataset: creates ordered windows and supervised labels.
#   - model_forecast: converts a complete time series into model-ready windows and returns predictions.
#   - build_model: constructs a Conv1D + LSTM model with configurable architecture.
#   - evaluate_forecast: converts predictions back to the original scale and computes MAE, MSE, MAPE, and correlation.
# ==============================================================================

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer, seed=SEED)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def model_forecast(model, series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    forecast = model.predict(ds, verbose=0)
    return forecast[:, 0]

def build_model(filters=32, kernel_size=5, lstm_units=64, dense1=30, dense2=10, learning_rate=1e-3):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=[None, 1]),
        tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1,
                               padding="causal", activation="relu"),
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dense(dense1, activation="relu"),
        tf.keras.layers.Dense(dense2, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    return model

def evaluate_forecast(actual_raw, pred_norm, mean, std):
    pred_raw = pred_norm * std + mean
    mae = np.mean(np.abs(actual_raw - pred_raw))
    mse = np.mean(np.square(actual_raw - pred_raw))
    mape = np.mean(np.abs((actual_raw - pred_raw) / actual_raw)) * 100
    corr = np.corrcoef(actual_raw, pred_raw)[0, 1]
    return pred_raw, {"MAE": mae, "MSE": mse, "MAPE (%)": mape, "Correlation": corr}

# ==============================================================================
# 7. Baseline forecast
# The naive baseline uses the last observed value as the next forecast. For stock indices this is often a very strong benchmark.
# ==============================================================================

naive_forecast = series[split_time - 1:-1]
naive_metrics = {
    "Experiment": "Naive forecast",
    "window_size": 1,
    "batch_size": "—",
    "shuffle_buffer_size": "—",
    "epochs": "—",
    "filters": "—",
    "kernel_size": "—",
    "lstm_units": "—",
    "Dense": "—",
    "MAE": np.mean(np.abs(x_valid_raw - naive_forecast)),
    "MSE": np.mean((x_valid_raw - naive_forecast) ** 2),
    "MAPE (%)": np.mean(np.abs((x_valid_raw - naive_forecast) / x_valid_raw)) * 100,
    "Correlation": np.corrcoef(x_valid_raw, naive_forecast)[0, 1],
}
naive_metrics

# ==============================================================================
# 8. Experiments
# The first three experiments adjust training parameters (window_size, batch_size, shuffle_buffer_size, epochs).
# The last two experiments adjust model architecture (filters, kernel_size, LSTM units, and Dense layer sizes).
# ==============================================================================

experiments = [
    dict(name="Baseline CNN-LSTM", window_size=30, batch_size=32, shuffle_buffer_size=1000,
         epochs=50, filters=32, kernel_size=5, lstm_units=64, dense1=30, dense2=10),
    dict(name="Short window", window_size=14, batch_size=16, shuffle_buffer_size=500,
         epochs=30, filters=32, kernel_size=5, lstm_units=64, dense1=30, dense2=10),
    dict(name="Long window", window_size=60, batch_size=64, shuffle_buffer_size=1500,
         epochs=70, filters=32, kernel_size=5, lstm_units=64, dense1=30, dense2=10),
    dict(name="Small architecture", window_size=30, batch_size=32, shuffle_buffer_size=1000,
         epochs=50, filters=16, kernel_size=3, lstm_units=32, dense1=16, dense2=8),
    dict(name="Large architecture", window_size=30, batch_size=32, shuffle_buffer_size=1000,
         epochs=70, filters=64, kernel_size=7, lstm_units=96, dense1=64, dense2=16),
]

histories = {}
forecasts = {}
rows = [naive_metrics]

for cfg in experiments:
    print("\nTraining:", cfg["name"])
    train_set = windowed_dataset(x_train, cfg["window_size"], cfg["batch_size"], cfg["shuffle_buffer_size"])
    model = build_model(filters=cfg["filters"], kernel_size=cfg["kernel_size"],
                        lstm_units=cfg["lstm_units"], dense1=cfg["dense1"], dense2=cfg["dense2"])
    if cfg["name"] == "Baseline CNN-LSTM":
        model.summary()
    history = model.fit(train_set, epochs=cfg["epochs"], verbose=0)
    histories[cfg["name"]] = history.history

    forecast_norm = model_forecast(model, series_norm, cfg["window_size"])
    # Align: prediction at position t uses window ending at t-1
    forecast_norm_valid = forecast_norm[split_time - cfg["window_size"] : -1]
    pred_raw, metric_values = evaluate_forecast(x_valid_raw, forecast_norm_valid, train_mean, train_std)
    forecasts[cfg["name"]] = pred_raw

    rows.append({
        "Experiment": cfg["name"],
        "window_size": cfg["window_size"],
        "batch_size": cfg["batch_size"],
        "shuffle_buffer_size": cfg["shuffle_buffer_size"],
        "epochs": cfg["epochs"],
        "filters": cfg["filters"],
        "kernel_size": cfg["kernel_size"],
        "lstm_units": cfg["lstm_units"],
        "Dense": f'{cfg["dense1"]}/{cfg["dense2"]}',
        **metric_values
    })

results_df = pd.DataFrame(rows)
print(results_df.to_string(index=False))

# ==============================================================================
# 9. Results from the accompanying report run
# These results are included so the report remains complete even before rerunning all Colab training cells.
# When you run the notebook, your numbers may differ slightly because neural-network training is stochastic.
# ==============================================================================

reported_results = pd.DataFrame([
  {
    "Experiment": "Naive forecast",
    "window_size": 1,
    "batch_size": "\u2014",
    "shuffle_buffer_size": "\u2014",
    "epochs": "\u2014",
    "filters": "\u2014",
    "kernel_size": "\u2014",
    "lstm_units": "\u2014",
    "Dense": "\u2014",
    "MAE": 48.212688172042995,
    "MSE": 3967.765481720428,
    "MAPE (%)": 1.0951273131866461,
    "Correlation": 0.9969162899744715
  },
  {
    "Experiment": "Baseline CNN-LSTM",
    "window_size": 30,
    "batch_size": 32,
    "shuffle_buffer_size": 1000,
    "epochs": 80,
    "filters": 32,
    "kernel_size": 5,
    "lstm_units": 64,
    "Dense": "30/10",
    "MAE": 120.76189718646378,
    "MSE": 22635.816446460034,
    "MAPE (%)": 2.6671889328087883,
    "Correlation": 0.9938497537702896
  },
  {
    "Experiment": "Short window",
    "window_size": 14,
    "batch_size": 16,
    "shuffle_buffer_size": 500,
    "epochs": 60,
    "filters": 32,
    "kernel_size": 5,
    "lstm_units": 64,
    "Dense": "30/10",
    "MAE": 62.491219498672656,
    "MSE": 6334.267371439868,
    "MAPE (%)": 1.424175678340029,
    "Correlation": 0.9952832842731099
  },
  {
    "Experiment": "Long window",
    "window_size": 60,
    "batch_size": 64,
    "shuffle_buffer_size": 1500,
    "epochs": 100,
    "filters": 32,
    "kernel_size": 5,
    "lstm_units": 64,
    "Dense": "30/10",
    "MAE": 154.1899401380767,
    "MSE": 35641.14259610642,
    "MAPE (%)": 3.3590394584350474,
    "Correlation": 0.9931849591897244
  },
  {
    "Experiment": "Small architecture",
    "window_size": 30,
    "batch_size": 32,
    "shuffle_buffer_size": 1000,
    "epochs": 80,
    "filters": 16,
    "kernel_size": 3,
    "lstm_units": 32,
    "Dense": "16/8",
    "MAE": 100.53340187063581,
    "MSE": 15470.906428734312,
    "MAPE (%)": 2.2025492028914866,
    "Correlation": 0.9911072941100804
  },
  {
    "Experiment": "Large architecture",
    "window_size": 30,
    "batch_size": 32,
    "shuffle_buffer_size": 1000,
    "epochs": 100,
    "filters": 64,
    "kernel_size": 7,
    "lstm_units": 96,
    "Dense": "64/16",
    "MAE": 127.33640202391757,
    "MSE": 25956.64905167306,
    "MAPE (%)": 2.8097655109983353,
    "Correlation": 0.9934996859456442
  }
])
print(reported_results.to_string(index=False))

# ==============================================================================
# 10. Training history and forecast plots
# ==============================================================================

# Pick best neural model by validation MAE
neural_results = results_df[results_df["Experiment"] != "Naive forecast"].copy()
best_name = neural_results.sort_values("MAE").iloc[0]["Experiment"]
print("Best neural model:", best_name)

plt.figure(figsize=(10, 5))
for name, hist in histories.items():
    plt.plot(hist["loss"], label=name)
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Huber loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plot_series(time_valid, x_valid_raw, label="Actual")
plot_series(time_valid, forecasts[best_name], label=f"Forecast: {best_name}")
plot_series(time_valid, naive_forecast, label="Naive forecast")
plt.title("Forecast comparison on validation set")
plt.legend()
plt.show()

# Illustrative ±10% interval, not a statistical confidence interval
pred = forecasts[best_name]
lower = pred * 0.90
upper = pred * 1.10

plt.figure(figsize=(12, 5))
plt.plot(time_valid, x_valid_raw, label="Actual")
plt.plot(time_valid, pred, label=f"Forecast: {best_name}")
plt.fill_between(time_valid, lower, upper, alpha=0.2, label="Illustrative ±10% band")
plt.title("Forecast with illustrative error band")
plt.xlabel("Observation")
plt.ylabel("DAX")
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# 11. Model evaluation plots
# Residual histograms, regression plots, Taylor diagrams, and Bland–Altman plots give complementary views of error behavior.
# ==============================================================================

best_pred = forecasts[best_name]
residuals = x_valid_raw - best_pred

# Error histogram
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=28)
plt.axvline(np.mean(residuals), linestyle="--", label=f"Mean residual = {np.mean(residuals):.2f}")
plt.title("Error histogram")
plt.xlabel("Actual - predicted")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()

# Regression plot
plt.figure(figsize=(6, 6))
plt.scatter(x_valid_raw, best_pred, alpha=0.55, s=16)
mn = min(x_valid_raw.min(), best_pred.min())
mx = max(x_valid_raw.max(), best_pred.max())
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("Regression plot")
plt.grid(True)
plt.show()

# Taylor diagram without extra packages
def taylor_diagram(actual, model_outputs, labels):
    actual = np.asarray(actual)
    actual_centered = actual - actual.mean()
    std_obs = actual_centered.std(ddof=0)
    max_std = max([std_obs] + [np.std(np.asarray(y) - np.mean(y)) for y in model_outputs]) * 1.25

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(0, std_obs, "o", label="Observation")

    for y, label in zip(model_outputs, labels):
        y_centered = np.asarray(y) - np.mean(y)
        corr = np.corrcoef(actual_centered, y_centered)[0, 1]
        theta = np.arccos(np.clip(corr, -1, 1))
        ax.plot(theta, y_centered.std(ddof=0), "o", label=f"{label} (r={corr:.3f})")

    corrs = np.array([0, .2, .4, .6, .8, .9, .95, .99, 1.0])
    ax.set_thetagrids(np.degrees(np.arccos(corrs)), labels=[f"{c:.2g}" for c in corrs])
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(0, max_std)
    ax.set_title("Taylor diagram")
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.show()

taylor_diagram(x_valid_raw, [best_pred, naive_forecast], [best_name, "Naive"])

# Bland–Altman plot
avg = (x_valid_raw + best_pred) / 2
diff = x_valid_raw - best_pred
mean_diff = diff.mean()
std_diff = diff.std()
upper = mean_diff + 1.96 * std_diff
lower = mean_diff - 1.96 * std_diff

plt.figure(figsize=(9, 5))
plt.scatter(avg, diff, alpha=0.5, s=15)
plt.axhline(mean_diff, linestyle="--", label=f"Mean: {mean_diff:.2f}")
plt.axhline(upper, linestyle="--", label=f"+1.96 SD: {upper:.2f}")
plt.axhline(lower, linestyle="--", label=f"-1.96 SD: {lower:.2f}")
plt.title("Bland–Altman plot")
plt.xlabel("Mean of actual and predicted")
plt.ylabel("Actual - predicted")
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# 12. Conclusions
#   - The DAX series has strong trend and persistence, but no stable classical seasonal/cyclical pattern from decomposition.
#   - Smaller windows performed better than longer windows in the report run because the stock index is close to a random-walk process; recent values are most informative.
#   - Increasing architecture size did not guarantee better validation accuracy; larger models can overfit noisy financial movements.
#   - The naive forecast is a critical benchmark for stock-price forecasting. In this run it outperformed the neural variants, which is a realistic result for one-step-ahead stock index prediction.
#   - Leakage prevention was handled by chronological splitting, computing normalization only on training data, and generating validation forecasts using only past windows.
# ==============================================================================
