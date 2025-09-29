# EDA for Time Series Data — Framework

> A thorough, repeatable exploratory data analysis (EDA) checklist specifically for time series datasets. Use this as a template to inspect, visualize, clean, and prepare series for modeling.

---

## Table of contents

1. Goals & scope
2. Quick setup (imports & helper functions)
3. Load & initial inspection
4. Datetime handling & index
5. Basic summary statistics
6. Missing values: detection & strategies
7. Resampling & aggregation
8. Visualization (core plots)
9. Decomposition (trend/seasonality/residual)
10. Stationarity tests
11. Auto-correlation & PACF
12. Lag / scatter / season plots
13. Outliers & anomalies
14. Feature engineering for models
15. Cross-correlation & multivariate checks
16. Spectral analysis (periodicity)
17. Preparing train/val/test splits
18. Pipelines, scaling & leakage prevention
19. Deliverables & EDA checklist
20. Common pitfalls & tips

---

# 1. Goals & scope

* Understand data frequency, coverage, and quality.
* Identify trend, seasonality, cycles, and anomalies.
* Decide resampling/aggregation level and imputation strategy.
* Create features and test stationarity for modeling.

---

# 2. Quick setup (imports & helper functions)

```python
# minimal imports for EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal
from sklearn.ensemble import IsolationForest

plt.rcParams['figure.figsize'] = (12, 4)

# helper: quick line plot
def quick_plot(series, title=None):
    plt.figure()
    series.plot()
    if title: plt.title(title)
    plt.tight_layout()
```

---

# 3. Load & initial inspection

* Load with `parse_dates` for date columns.
* Inspect `df.head()`, `df.info()`, `df.describe()`.

```python
df = pd.read_csv('data.csv', parse_dates=['timestamp'])
print(df.info())
df.head()
```

Key checks: missing timestamp rows, duplicate timestamps, unexpected dtypes.

---

# 4. Datetime handling & index

* Convert timestamp column to `DatetimeIndex` and set frequency if possible.
* Check for duplicates and sort.

```python
df = df.sort_values('timestamp').drop_duplicates('timestamp')
df = df.set_index('timestamp')
# infer frequency (may return None)
print(pd.infer_freq(df.index))
# if regular: df = df.asfreq('H')
```

If `infer_freq` is None, examine diffs `df.index.to_series().diff().value_counts()`.

---

# 5. Basic summary statistics

* Use `.describe()` on target; inspect distribution, min/max, percentiles.
* Check percent of constant or near-constant values.

```python
print(df['y'].describe())
print('unique rate:', df['y'].nunique() / len(df))
```

---

# 6. Missing values: detection & strategies

* Visualize missingness over time (heatmap or vertical bars).
* Strategy depends on gap length & downstream model: forward/backward fill, interpolation, model-based imputation, or leave-as-missing for models that accept NaN.

```python
# percent missing
print(df.isna().mean())
# visualize missing by resampling
missing = df['y'].isna().astype(int).resample('D').sum()
missing.plot(kind='bar');

# short gaps: interpolation
df['y_interp'] = df['y'].interpolate(method='time')
# long gaps: flag and avoid imputing
```

---

# 7. Resampling & aggregation

* Resample to desired granularity (hourly, daily, weekly). Compare results with original frequency.

```python
# daily sum vs mean
daily = df['y'].resample('D').sum()
daily.plot();
```

---

# 8. Visualization (core plots)

* Line plot (full range), zoomed windows, seasonal subplots (by year/month/day), histogram & KDE, boxplots by period, rolling mean/std.

```python
quick_plot(df['y'], 'Full series')
# rolling
df['y'].rolling(24).mean().plot(label='24h mean')
df['y'].rolling(24).std().plot(label='24h std')
plt.legend();

# boxplot by month
df['month'] = df.index.month
df.boxplot(column='y', by='month')
```

---

# 9. Decomposition (trend / seasonality / residual)

* Apply additive and multiplicative decomposition depending on variance behaviour.

```python
res = seasonal_decompose(df['y'].dropna(), model='additive', period=24)
res.plot();
```

Interpret residuals: are they stationary? contain structure?

---

# 10. Stationarity tests

* ADF (null: non-stationary), KPSS (null: stationary). Use both for robust decision.

```python
def adf_test(x):
    stat, p, *_ = adfuller(x.dropna())
    print('ADF stat:', stat, 'p-value:', p)

def kpss_test(x):
    stat, p, *_ = kpss(x.dropna(), regression='c')
    print('KPSS stat:', stat, 'p-value:', p)

adf_test(df['y'])
kpss_test(df['y'])
```

If non-stationary, consider differencing (seasonal &/or regular) or detrending.

---

# 11. Auto-correlation & PACF

* Use ACF/PACF plots to identify lag structure and AR/MA terms.

```python
plot_acf(df['y'].dropna(), lags=48);
plot_pacf(df['y'].dropna(), lags=48);
```

---

# 12. Lag plots & scatter by lag

* Visual check for autocorrelation and non-linear relationships.

```python
pd.plotting.lag_plot(df['y'], lag=1)
```

---

# 13. Outlier detection & robust checks

* Simple: IQR or z-score on residuals or differenced series.
* Model-based: IsolationForest on sliding-window features.

```python
# z-score on differenced series
d = df['y'].diff().dropna()
z = (d - d.mean())/d.std()
outliers = z.abs() > 3

# IsolationForest example on features
clf = IsolationForest(contamination=0.01)
feat = df['y'].fillna(method='ffill').to_frame()
feat['lag1'] = feat['y'].shift(1)
feat = feat.dropna()
clf.fit(feat)
feat['anomaly'] = clf.predict(feat)
```

---

# 14. Feature engineering for models

* Lags (y\_t-1..y\_t-n), rolling stats (mean/std/min/max), difference features, time features (hour, day, month, is\_weekend), cyclical transforms (sin/cos for hour/day), Fourier features for periodicity.

```python
# lag features
for l in [1,24,168]:
    df[f'lag_{l}'] = df['y'].shift(l)
# rolling
df['roll_24_mean'] = df['y'].rolling(24).mean()
# cyclical hour
df['hour'] = df.index.hour
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
```

---

# 15. Cross-correlation & multivariate checks

* If multiple series present, compute cross-correlation (CCF) and Granger causality as needed.

```python
# cross-correlation (simple)
from statsmodels.tsa.stattools import ccf
cc = ccf(df['x'].dropna(), df['y'].dropna())[:50]
plt.bar(range(len(cc)), cc)
```

---

# 16. Spectral analysis (periodicity)

* Use periodogram or FFT to find dominant frequencies.

```python
f, Pxx = signal.periodogram(df['y'].dropna())
plt.semilogy(f, Pxx)
plt.xlabel('Frequency')
```

---

# 17. Preparing train / validation / test splits (time-aware)

* Never random-split. Use contiguous splits by time.
* Consider time-based cross-validation (rolling-origin / expanding window) for model selection.

```python
def time_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

train, val, test = time_split(df.dropna())
```

---

# 18. Pipelines, scaling & leakage prevention

* Fit scalers and encoders on training data only.
* Keep pipeline code that transforms new data with the same fitted objects.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train[feat_cols])
train_scaled = scaler.transform(train[feat_cols])
val_scaled = scaler.transform(val[feat_cols])
```

Use `sklearn.pipeline.Pipeline` for non-time-aware preprocessing; for time series you may need custom transformers that respect index and ordering.

---

# 19. Deliverables & EDA checklist (what to include in a report)

* Summary: data range, frequency, missingness, duplicate timestamps.
* Plots: full-series + zoom, seasonal plots, decomposition, ACF/PACF, histogram, boxplots.
* Stationarity test results (ADF/KPSS) with interpretation.
* Imputation & resampling decisions with rationale.
* Feature list & brief rationale.
* Train/val/test split with exact date cutoffs.
* Any identified anomalies with example timestamps.

---

# 20. Common pitfalls & tips

* **Data leakage**: never use future information to create features for past rows.
* **Frequency mismatch**: mixing multiple frequencies without resampling leads to bugs.
* **Silent timezone issues**: convert all timestamps to a common timezone or UTC.
* **Using mean imputation blindly**: can remove seasonality; prefer time-aware interpolation or model-based imputation.
* **Ignoring non-stationary residuals**: residual structure means your model is missing patterns.

---

# 21. Next steps / suggestions

* Convert this EDA into a reproducible Jupyter notebook with: data loading, plots, and saved artifacts (plots + CSVs of features).
* Add unit tests for feature creation (e.g., shapes, no future leakage).
* If modeling, start with simple baselines: naive (last value), seasonal naive, exponential smoothing, then progress to ARIMA/Prophet/Gradient boosting / deep learning.

---

*This file is intended to be a complete starting point. If you want, I can:*

* convert this into a runnable Jupyter notebook with the same code cells, or
* expand any single section into a more detailed tutorial (e.g., feature engineering or seasonality detection).
