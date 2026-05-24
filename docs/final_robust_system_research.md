# Robust Generic Forecasting System: Complete Research & Implementation Guide

> **Scope**: How to build a forecasting system that works for ANY data, ANY year, ANY schema - not just the test data.  
> **Date**: Research compiled from 20+ web searches across authoritative sources.  
> **Sources**: Nixtla/StatsForecast, Darts, Prophet, Pandas, scikit-learn, MLflow, Pandera, Evidently AI, CIKM'22 AutoForecast paper, and more.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Auto-Detect Time Series Frequency](#3-auto-detect-time-series-frequency)
4. [Automatic Schema Detection](#4-automatic-schema-detection)
5. [Data Validation & Quality Checks](#5-data-validation--quality-checks)
6. [Configuration-Driven Design](#6-configuration-driven-design)
7. [Automatic Feature Engineering](#7-automatic-feature-engineering)
8. [Generic Holiday Handling (Any Year)](#8-generic-holiday-handling-any-year)
9. [Automatic Model Selection](#9-automatic-model-selection)
10. [Robust Cross-Validation](#10-robust-cross-validation)
11. [Fallback Strategies](#11-fallback-strategies)
12. [Production System Architecture](#12-production-system-architecture)
13. [MLflow Integration & Model Registry](#13-mlflow-integration--model-registry)
14. [Complete Implementation Code](#14-complete-implementation-code)
15. [Key Recommendations](#15-key-recommendations)

---

## 1. Executive Summary

### The Core Problem

A forecasting system that works for "any data" must solve these challenges:

| Challenge | Why It Breaks Systems | Solution |
|---|---|---|
| Different column names | Hard-coded `date`/`sales` columns fail | Auto-detect date/value columns |
| Different number of customers/products | Fixed-width features break | Dynamic groupby + iteration |
| Different date ranges | Assumed start/end dates fail | Auto-detect date range from data |
| Different year (not 2026) | Hard-coded 2026 holidays fail | Dynamic holiday lookup by year |
| Missing values | NaN crashes models | Auto-fill with forward/backward fill |
| Different file formats | Only CSV support fails | Support CSV, Excel, Parquet, JSON |
| Insufficient data | Model training fails | Naive fallback strategies |
| Different frequencies | Daily model on monthly data | Auto-detect `D`/`W`/`M`/`Y` |

### The Robust System Blueprint

```
INPUT (Any Format/Schema) 
  --> Schema Detector (auto-detect columns, types, frequency)
  --> Data Validator (Pandera/Great Expectations checks)
  --> Preprocessor (fill missing, resample, regularize)
  --> Feature Engineer (auto lags, rolling stats, holidays)
  --> Model Selector (try multiple, pick best via CV)
  --> Fallback Handler (naive if data insufficient)
  --> Output (forecasts for any future period)
```

---

## 2. Architecture Overview

### 2.1 Layered Architecture (Production-Grade)

```
┌─────────────────────────────────────────────────────────────┐
│                    CONFIG LAYER (YAML/JSON)                  │
│  column_mappings, model_params, holidays, thresholds         │
├─────────────────────────────────────────────────────────────┤
│                    DATA INGESTION LAYER                      │
│  CSV/Excel/Parquet/JSON --> Pandas --> Schema Detection      │
├─────────────────────────────────────────────────────────────┤
│                    VALIDATION LAYER                          │
│  Pandera schema checks, data quality rules, drift detection  │
├─────────────────────────────────────────────────────────────┤
│                    PREPROCESSING LAYER                       │
│  Missing value imputation, frequency inference, resampling   │
├─────────────────────────────────────────────────────────────┤
│                    FEATURE ENGINEERING LAYER                 │
│  Auto lags, rolling stats, date features, holidays, Fourier  │
├─────────────────────────────────────────────────────────────┤
│                    MODELING LAYER                            │
│  Multiple models (AutoARIMA, ETS, ML, Naive) + CV selection│
├─────────────────────────────────────────────────────────────┤
│                    FALLBACK LAYER                            │
│  Naive forecast, seasonal naive, mean forecast, zero         │
├─────────────────────────────────────────────────────────────┤
│                    OUTPUT LAYER                              │
│  Forecasts + confidence intervals + model metadata           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Why Configuration-Driven?

> "Instead of hard-coding these values into Python scripts, which can lead to messy codebases, you separate this data into a YAML file. This separation of concerns allows researchers to swap datasets or adjust learning rates without touching the core codebase." - [YAML Configuration AI Systems](https://www.guild.ai/glossary/yaml-configuration-ai-systems)

**Benefits:**
- **No code changes** when data schema changes
- **Version-controllable** config files (git-tracked)
- **Environment-specific** configs (dev/staging/prod)
- **Reproducible** experiments via config snapshots
- **Non-technical users** can adjust parameters

---

## 3. Auto-Detect Time Series Frequency

### 3.1 Pandas `infer_freq()`

Pandas provides a built-in function to detect frequency automatically [^1438^][^1439^][^1445^]:

```python
import pandas as pd
import numpy as np

def infer_frequency(df, date_column):
    """
    Infer the frequency of a time series DataFrame.
    Returns: 'D' (daily), 'W' (weekly), 'ME' (monthly), 
             'YE' (yearly), 'H' (hourly), or None if irregular.
    """
    # Ensure datetime index
    if date_column in df.columns:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column).sort_index()
    
    # Try pandas infer_freq
    freq = pd.infer_freq(df.index)
    
    if freq is None:
        # Fallback: calculate median difference between consecutive dates
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            freq = 'H'
        elif median_diff <= pd.Timedelta(days=2):
            freq = 'D'
        elif median_diff <= pd.Timedelta(days=8):
            freq = 'W'
        elif median_diff <= pd.Timedelta(days=35):
            freq = 'ME'
        elif median_diff <= pd.Timedelta(days=370):
            freq = 'YE'
        else:
            freq = 'D'  # ultimate fallback
    
    # Normalize aliases
    freq_map = {
        'M': 'ME', 'MS': 'ME', 'ME': 'ME',
        'W-MON': 'W', 'W-TUE': 'W', 'W': 'W',
        'A': 'YE', 'Y': 'YE', 'YE': 'YE', 'YS': 'YE',
        'B': 'D', 'C': 'D',
    }
    return freq_map.get(freq, freq)
```

### 3.2 Making Irregular Time Series Regular [^1556^][^1559^]

```python
def regularize_time_series(df, date_col, value_cols, freq=None):
    """
    Convert irregular time series to regular frequency.
    Fills gaps with forward fill + backward fill.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Auto-detect frequency if not provided
    if freq is None:
        freq = infer_frequency(df.reset_index(), df.index.name or 'date')
    
    # Create complete date range
    full_range = pd.date_range(start=df.index.min(), 
                               end=df.index.max(), 
                               freq=freq)
    
    # Reindex to fill gaps
    df_regular = df.reindex(full_range)
    df_regular.index.name = date_col
    
    # Fill strategy: forward fill, then backward fill for leading NaNs
    df_regular[value_cols] = df_regular[value_cols].ffill().bfill()
    
    return df_regular.reset_index(), freq
```

---

## 4. Automatic Schema Detection

### 4.1 Auto-Detect Date Column

```python
import pandas as pd

def detect_date_column(df, hint=None):
    """
    Automatically detect which column contains dates.
    Checks column names, then attempts to parse each column.
    """
    date_keywords = ['date', 'time', 'day', 'month', 'year', 'period', 
                     'timestamp', 'ds', 'dt', 'fecha', 'datum', 'jour']
    
    # If hint provided, try that first
    if hint and hint in df.columns:
        return hint
    
    # Check column names for date keywords
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in date_keywords):
            try:
                pd.to_datetime(df[col], errors='raise')
                return col
            except (ValueError, TypeError):
                continue
    
    # Try parsing each column as datetime
    for col in df.columns:
        try:
            # Try converting first non-null value
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample is not None:
                pd.to_datetime(sample)
                # Check if majority of values can be parsed as dates
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() / len(df) > 0.8:  # 80% parseable
                    return col
        except (ValueError, TypeError, IndexError):
            continue
    
    raise ValueError(f"Could not auto-detect date column. Columns: {list(df.columns)}")
```

### 4.2 Auto-Detect Value/Target Columns

```python
def detect_value_columns(df, date_col, id_col=None):
    """
    Automatically detect which columns are numeric values (not IDs/dates).
    Returns list of column names that are numeric and likely targets.
    """
    exclude = [date_col]
    if id_col:
        exclude.append(id_col)
    
    # Exclude obvious ID columns
    id_keywords = ['id', 'key', 'code', 'sku', 'name', 'category', 'type']
    value_cols = []
    
    for col in df.columns:
        if col in exclude:
            continue
        
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in id_keywords):
            continue
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            value_cols.append(col)
    
    if not value_cols:
        # Fallback: return first numeric column that's not in exclude
        for col in df.columns:
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
                value_cols.append(col)
    
    return value_cols
```

### 4.3 Auto-Detect ID Column (for multiple series)

```python
def detect_id_column(df, date_col):
    """
    Detect column that identifies different time series.
    Looks for columns with moderate cardinality (not unique per row, not constant).
    """
    best_col = None
    best_score = -1
    
    for col in df.columns:
        if col == date_col:
            continue
        
        n_unique = df[col].nunique()
        n_rows = len(df)
        
        # Good ID columns: between 2 and sqrt(n_rows) unique values
        if 2 <= n_unique <= min(10000, int(n_rows ** 0.5)):
            # Prefer string or integer types
            if pd.api.types.is_string_dtype(df[col]) or \
               pd.api.types.is_integer_dtype(df[col]):
                score = min(n_unique, 1000)  # Prefer moderate cardinality
                if score > best_score:
                    best_score = score
                    best_col = col
    
    return best_col  # None means single time series
```

### 4.4 Complete Schema Detector

```python
class SchemaDetector:
    """
    Automatically detect the schema of any time series dataset.
    Works with any column names, any number of series, any date range.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.date_col = None
        self.id_col = None
        self.value_cols = None
        self.freq = None
    
    def detect(self, df):
        """Run full schema detection."""
        # Step 1: Detect date column
        self.date_col = self.config.get('date_column') or detect_date_column(df)
        
        # Step 2: Detect ID column
        self.id_col = self.config.get('id_column') or detect_id_column(df, self.date_col)
        
        # Step 3: Detect value columns
        self.value_cols = self.config.get('value_columns') or \
                          detect_value_columns(df, self.date_col, self.id_col)
        
        # Step 4: Detect frequency
        sample_df = df if self.id_col is None else df[df[self.id_col] == df[self.id_col].iloc[0]]
        self.freq = infer_frequency(sample_df, self.date_col)
        
        return {
            'date_column': self.date_col,
            'id_column': self.id_col,
            'value_columns': self.value_cols,
            'frequency': self.freq,
            'date_range': (df[self.date_col].min(), df[self.date_col].max()),
            'n_series': df[self.id_col].nunique() if self.id_col else 1,
            'n_rows': len(df),
        }
```

---

## 5. Data Validation & Quality Checks

### 5.1 Pandera for DataFrame Validation [^1539^][^1544^][^1547^][^1548^]

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check

def create_time_series_schema(date_col, value_cols, freq='D'):
    """
    Create a Pandera schema for time series data validation.
    Automatically validates any column names.
    """
    columns = {
        date_col: Column(pa.DateTime, nullable=False),
    }
    
    for col in value_cols:
        columns[col] = Column(
            pa.Float,  # Coerce to float to handle ints too
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(-1e12),  # Sanity bounds
                Check.less_than_or_equal_to(1e12),
            ]
        )
    
    return DataFrameSchema(columns, strict=False)


def validate_dataframe(df, schema):
    """Validate dataframe against schema with error collection."""
    try:
        validated = schema.validate(df, lazy=True)
        return validated, []
    except pa.errors.SchemaErrors as err:
        return df, err.failure_cases.to_dict('records')
```

### 5.2 Automated Data Quality Checks [^1564^][^1552^]

```python
class DataQualityChecker:
    """
    Automated data quality checks for any time series dataset.
    Inspired by Write-Audit-Publish pattern.
    """
    
    def __init__(self):
        self.issues = []
    
    def check(self, df, date_col, value_cols, id_col=None):
        """Run all quality checks. Returns True if data passes."""
        self.issues = []
        
        # 1. Check for missing values
        for col in value_cols:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0.5:
                self.issues.append({
                    'severity': 'FAIL',
                    'check': 'missing_values',
                    'column': col,
                    'message': f'{missing_pct:.1%} missing values (>50%)'
                })
            elif missing_pct > 0.05:
                self.issues.append({
                    'severity': 'WARN',
                    'check': 'missing_values',
                    'column': col,
                    'message': f'{missing_pct:.1%} missing values'
                })
        
        # 2. Check for duplicates
        check_cols = [date_col] + ([id_col] if id_col else [])
        n_dups = df.duplicated(subset=check_cols).sum()
        if n_dups > 0:
            self.issues.append({
                'severity': 'WARN',
                'check': 'duplicates',
                'message': f'{n_dups} duplicate rows found'
            })
        
        # 3. Check for negative values (sales should be non-negative)
        for col in value_cols:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                self.issues.append({
                    'severity': 'WARN',
                    'check': 'negative_values',
                    'column': col,
                    'message': f'{n_negative} negative values found'
                })
        
        # 4. Check for sufficient data length
        min_required = 14  # At least 14 observations
        if id_col:
            series_lengths = df.groupby(id_col).size()
            short_series = (series_lengths < min_required).sum()
            if short_series > 0:
                self.issues.append({
                    'severity': 'WARN',
                    'check': 'insufficient_data',
                    'message': f'{short_series} series have < {min_required} observations'
                })
        else:
            if len(df) < min_required:
                self.issues.append({
                    'severity': 'FAIL',
                    'check': 'insufficient_data',
                    'message': f'Only {len(df)} observations (need {min_required}+)'
                })
        
        # 5. Check date range
        date_span = (df[date_col].max() - df[date_col].min()).days
        if date_span < 30:
            self.issues.append({
                'severity': 'WARN',
                'check': 'short_history',
                'message': f'Data spans only {date_span} days'
            })
        
        return len([i for i in self.issues if i['severity'] == 'FAIL']) == 0
    
    def get_issues(self):
        return self.issues
```

### 5.3 Schema Drift Detection [^1547^]

```python
def detect_schema_drift(new_df, expected_schema):
    """
    Detect when incoming data has different schema than expected.
    Useful for production monitoring.
    """
    drift = []
    
    # Check for missing columns
    for col in expected_schema['columns']:
        if col not in new_df.columns:
            drift.append({'type': 'missing_column', 'column': col})
    
    # Check for new columns
    for col in new_df.columns:
        if col not in expected_schema['columns']:
            drift.append({'type': 'new_column', 'column': col})
    
    # Check type changes
    for col, expected_type in expected_schema.get('dtypes', {}).items():
        if col in new_df.columns:
            actual_type = str(new_df[col].dtype)
            if actual_type != expected_type:
                drift.append({
                    'type': 'type_change',
                    'column': col,
                    'expected': expected_type,
                    'actual': actual_type
                })
    
    return drift
```

---

## 6. Configuration-Driven Design

### 6.1 YAML Configuration Structure

```yaml
# config.yaml - Generic forecasting system configuration
system:
  name: "generic_forecast"
  version: "1.0.0"
  random_seed: 42

data:
  # Auto-detect if null; override with specific column names
  date_column: null       # e.g., "date", "ds", "timestamp"
  id_column: null         # e.g., "customer_id", "product_id", "series_id"
  value_columns: null     # e.g., ["sales", "quantity"]
  frequency: null         # "D", "W", "ME", "YE" - auto-detect if null
  
  # File handling
  input_format: "auto"    # "csv", "excel", "parquet", "auto"
  encoding: "utf-8"
  date_format: null       # e.g., "%Y-%m-%d"

preprocessing:
  missing_value_strategy: "auto"  # "auto", "ffill", "interpolate", "mean"
  outlier_method: "iqr"           # "iqr", "zscore", "none"
  outlier_threshold: 3.0
  regularize: true                # Fill gaps in time series

features:
  # Automatic feature generation
  auto_lags: true
  lag_periods: "auto"             # "auto", or [1, 7, 14, 28]
  
  auto_rolling: true
  rolling_windows: "auto"         # "auto", or [7, 14, 30]
  rolling_stats: ["mean", "std", "min", "max"]
  
  date_features: true
  date_features_list: ["dayofweek", "month", "year", "quarter"]
  
  holidays:
    enabled: true
    country: "US"                 # "US", "UK", "CA", "DE", etc.
    state: null                   # e.g., "CA" for California
    years: "auto"                 # "auto" = detect from data

models:
  # Models to try (system will pick best via CV)
  candidates:
    - name: "auto_arima"
      enabled: true
      params:
        seasonal: true
    - name: "ets"
      enabled: true
    - name: "theta"
      enabled: true
    - name: "naive"
      enabled: true
    - name: "seasonal_naive"
      enabled: true
  
  # Model selection
  cv_folds: 3
  cv_horizon: "auto"              # "auto" = 1 season, or integer
  metric: "mae"                   # "mae", "rmse", "mape"

forecast:
  horizon: "auto"                 # "auto" = same as data frequency for 30 periods
  confidence_interval: 0.95
  
  # Fallback
  fallback:
    enabled: true
    min_observations: 14
    method: "naive"               # "naive", "mean", "zero", "last_value"

output:
  format: "csv"                   # "csv", "excel", "parquet"
  include_metadata: true
  include_feature_importance: true

logging:
  level: "INFO"
  log_file: "forecast.log"
```

### 6.2 Config Loader

```python
import yaml
import os

class Config:
    """Configuration loader with environment variable substitution."""
    
    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        config = cls._substitute_env(config)
        return config
    
    @classmethod
    def from_dict(cls, d):
        return d
    
    @staticmethod
    def _substitute_env(obj):
        """Recursively substitute ${VAR} with environment variables."""
        if isinstance(obj, dict):
            return {k: Config._substitute_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._substitute_env(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        return obj
    
    def get(self, path, default=None):
        """Get nested config value by dot path, e.g., 'models.cv_folds'."""
        keys = path.split('.')
        val = self._config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val
```

---

## 7. Automatic Feature Engineering

### 7.1 Lag Features [^1251^][^70^][^1470^]

```python
def create_lag_features(df, value_cols, id_col=None, lags=None):
    """
    Automatically create lag features.
    Uses smart defaults based on detected frequency.
    """
    df = df.copy().sort_values(by=[id_col] if id_col else []).reset_index(drop=True)
    
    if lags is None or lags == "auto":
        # Smart lag selection based on frequency
        lags = [1, 7, 14, 28]  # yesterday, week ago, 2 weeks, 4 weeks
    
    new_features = []
    groups = df.groupby(id_col) if id_col else [(None, df)]
    
    result_dfs = []
    for group_id, group_df in groups:
        group_df = group_df.copy()
        for col in value_cols:
            for lag in lags:
                feat_name = f'{col}_lag_{lag}'
                group_df[feat_name] = group_df[col].shift(lag)
                new_features.append(feat_name)
        result_dfs.append(group_df)
    
    return pd.concat(result_dfs, ignore_index=True), new_features
```

### 7.2 Rolling Window Statistics [^1251^][^70^]

```python
def create_rolling_features(df, value_cols, id_col=None, windows=None, stats=None):
    """
    Create rolling window statistics automatically.
    """
    df = df.copy()
    
    if windows is None or windows == "auto":
        windows = [7, 14, 30, 60]
    if stats is None:
        stats = ['mean', 'std', 'min', 'max']
    
    new_features = []
    groups = df.groupby(id_col) if id_col else [(None, df)]
    
    result_dfs = []
    for group_id, group_df in groups:
        group_df = group_df.copy().sort_index()
        for col in value_cols:
            for window in windows:
                for stat in stats:
                    feat_name = f'{col}_rolling_{stat}_{window}'
                    rolling = group_df[col].shift(1).rolling(window=window, min_periods=1)
                    
                    if stat == 'mean':
                        group_df[feat_name] = rolling.mean()
                    elif stat == 'std':
                        group_df[feat_name] = rolling.std()
                    elif stat == 'min':
                        group_df[feat_name] = rolling.min()
                    elif stat == 'max':
                        group_df[feat_name] = rolling.max()
                    
                    new_features.append(feat_name)
        result_dfs.append(group_df)
    
    return pd.concat(result_dfs), new_features
```

### 7.3 Date/Calendar Features

```python
def create_date_features(df, date_col):
    """
    Extract calendar features from date column.
    Works for any year, any date range.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    df['dayofweek'] = dt.dt.dayofweek       # 0=Monday, 6=Sunday
    df['dayofmonth'] = dt.dt.day
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['year'] = dt.dt.year
    df['weekofyear'] = dt.dt.isocalendar().week.astype(int)
    df['is_month_start'] = dt.dt.is_month_start.astype(int)
    df['is_month_end'] = dt.dt.is_month_end.astype(int)
    
    # Cyclical encoding for dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Days since start (trend feature)
    df['days_since_start'] = (dt - dt.min()).dt.days
    
    feature_cols = ['dayofweek', 'dayofmonth', 'month', 'quarter', 'year',
                    'weekofyear', 'is_month_start', 'is_month_end',
                    'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
                    'days_since_start']
    
    return df, feature_cols
```

### 7.4 Using MLForecast for Automated Feature Engineering [^70^][^1496^]

```python
# Using Nixtla's MLForecast for automatic feature engineering
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.lag_transforms import RollingMean, ExpandingMean
import lightgbm as lgb

def auto_feature_engineering_mlforecast(df, freq='D'):
    """
    Use MLForecast to automatically create features.
    Works with any frequency, handles multiple series.
    """
    fcst = MLForecast(
        models=[lgb.LGBMRegressor(verbosity=-1)],
        freq=freq,
        lags=[1, 7, 14, 28],
        lag_transforms={
            1: [RollingMean(window_size=7)],    # 7-day rolling mean
            7: [ExpandingMean()],                # Expanding mean of weekly
        },
        date_features=['dayofweek', 'month', 'year'],
    )
    
    # This preprocesses the data with all features
    features_df = fcst.preprocess(df)
    return features_df
```

### 7.5 Using TSFresh for Comprehensive Feature Extraction [^1479^][^1480^][^1481^]

```python
# TSFresh: Automatic extraction of 100s of features
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

def auto_features_tsfresh(df, id_col, date_col, value_col):
    """
    Use TSFresh for automatic feature extraction.
    Extracts hundreds of statistical, temporal, and spectral features.
    """
    # Extract features
    extracted = extract_features(
        df,
        column_id=id_col,
        column_sort=date_col,
        column_value=value_col,
        default_fc_parameters=EfficientFCParameters(),  # Balanced speed/coverage
        n_jobs=4
    )
    
    # Select relevant features (requires target)
    # y = ...  # your target variable
    # selected = select_features(extracted, y)
    
    return extracted
```

---

## 8. Generic Holiday Handling (Any Year)

### 8.1 Python Holidays Library [^1542^]

The `holidays` library supports **100+ countries** and generates holidays for **any year dynamically**:

```python
import holidays
import pandas as pd
from datetime import date

def get_holiday_features(df, date_col, country='US', state=None, years='auto'):
    """
    Create holiday features for ANY year automatically.
    Uses python-holidays library which supports 100+ countries.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Auto-detect years from data
    if years == 'auto':
        years = df[date_col].dt.year.unique().tolist()
        # Add forecast years too
        max_year = max(years)
        years.extend(range(max_year + 1, max_year + 3))
    
    # Create holiday calendar
    holiday_cal = holidays.country_holidays(
        country, 
        subdiv=state,
        years=years,
        observed=True  # Include observed holidays (e.g., Monday after Sunday)
    )
    
    # Check if each date is a holiday
    df['is_holiday'] = df[date_col].apply(
        lambda x: 1 if x.date() in holiday_cal else 0
    )
    
    # Get holiday name (empty string if not holiday)
    df['holiday_name'] = df[date_col].apply(
        lambda x: holiday_cal.get(x.date(), '')
    )
    
    # Days to/from nearest holiday
    holiday_dates = sorted([pd.Timestamp(d) for d in holiday_cal.keys()])
    
    df['days_to_holiday'] = df[date_col].apply(
        lambda x: min([(h - x).days for h in holiday_dates if h >= x], default=365)
    )
    df['days_from_holiday'] = df[date_col].apply(
        lambda x: min([(x - h).days for h in holiday_dates if h <= x], default=365)
    )
    
    # Weekend flag
    df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Month-end flag (common business pattern)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    
    return df
```

### 8.2 Multi-Country Holiday Support

```python
def get_supported_countries():
    """List all countries supported by the holidays library."""
    return holidays.list_supported_countries()

# Examples:
# holidays.country_holidays('US', years=2027)  # US holidays for 2027
# holidays.country_holidays('UK', years=2027)  # UK holidays for 2027
# holidays.country_holidays('DE', subdiv='BY', years=2027)  # Bavaria
# holidays.country_holidays('CA', subdiv='ON', years=2027)  # Ontario
```

### 8.3 Custom Business Holidays

```python
def add_custom_holidays(df, date_col, custom_dates):
    """
    Add custom business holidays (Black Friday, company events, etc.)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    custom_holidays_set = set(pd.to_datetime(custom_dates))
    df['is_custom_holiday'] = df[date_col].isin(custom_holidays_set).astype(int)
    
    return df
```

---

## 9. Automatic Model Selection

### 9.1 StatsForecast Auto Model Selection [^1431^][^1473^][^1475^][^1477^]

Nixtla's StatsForecast provides **20x faster** automatic model selection than pmdarima [^1477^]:

```python
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoTheta, AutoCES, 
    Naive, SeasonalNaive, HistoricAverage
)
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse

def automatic_model_selection(df, freq='D', season_length=None, horizon=30):
    """
    Try multiple models and select the best one via cross-validation.
    Works for any data frequency, any series length.
    """
    # Auto-detect season length
    if season_length is None:
        season_map = {'D': 7, 'W': 52, 'ME': 12, 'YE': 1, 'H': 24}
        season_length = season_map.get(freq, 7)
    
    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
        AutoCES(season_length=season_length),
        Naive(),
        SeasonalNaive(season_length=season_length),
        HistoricAverage(),
    ]
    
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1,  # Parallel processing
    )
    
    # Cross-validation to pick best model
    cv_df = sf.cross_validation(
        df=df,
        h=horizon,
        step_size=horizon,
        n_windows=3,  # 3-fold CV
    )
    
    # Evaluate: which model has lowest MAE?
    from utilsforecast.evaluation import evaluate
    evaluation = evaluate(cv_df, metrics=[mae, mse])
    
    # Select best model per series
    best_models = evaluation.loc[evaluation.groupby('unique_id')['mae'].idxmin()]
    
    return sf, best_models, evaluation
```

### 9.2 Meta-Learning Approach (AutoForecast) [^1433^][^1438^]

The AutoForecast paper from CIKM'22 demonstrates a **meta-learning** approach [^1433^]:

> "Given a new (unseen) dataset, AutoForecast automatically determines, using the meta-features and the meta-learners, the best forecasting model among a large space of models, without the need to train and evaluate any of the different forecasting models on this new dataset."

Key insight: Extract **meta-features** from the time series (length, trend, seasonality strength, noise level, autocorrelation, etc.) and use a pre-trained meta-learner to predict which model will perform best.

```python
def extract_meta_features(series):
    """
    Extract meta-features from a time series for model selection.
    Based on AutoForecast paper (CIKM'22).
    """
    from scipy import stats
    
    features = {}
    s = series.dropna()
    
    # Simple features
    features['length'] = len(s)
    features['mean'] = s.mean()
    features['std'] = s.std()
    features['cv'] = s.std() / s.mean() if s.mean() != 0 else 0
    features['skewness'] = stats.skew(s)
    features['kurtosis'] = stats.kurtosis(s)
    
    # Trend
    x = np.arange(len(s))
    slope, intercept, r_value, _, _ = stats.linregress(x, s)
    features['trend_slope'] = slope
    features['trend_r2'] = r_value ** 2
    
    # Autocorrelation
    features['autocorr_lag1'] = s.autocorr(lag=1) if len(s) > 1 else 0
    features['autocorr_lag7'] = s.autocorr(lag=7) if len(s) > 7 else 0
    
    # Seasonality (approximate via FFT)
    fft = np.fft.fft(s - s.mean())
    power = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(s))
    # Find dominant frequency (excluding DC component)
    dominant_idx = np.argmax(power[1:len(power)//2]) + 1
    features['seasonality_strength'] = power[dominant_idx] / power.sum()
    
    return features
```

### 9.3 Darts Unified Model Interface [^1538^][^1544^]

Darts provides a **unified fit()/predict() API** across ALL models [^1538^]:

```python
from darts import TimeSeries
from darts.models import (
    ExponentialSmoothing, ARIMA, Prophet,
    NBEATSModel, TFTModel, LightGBMModel
)
from darts.metrics import mae, rmse

def compare_models_darts(series, horizon):
    """
    Compare multiple models using Darts' unified API.
    Any model can be swapped in without changing the code.
    """
    train, val = series[:-horizon], series[-horizon:]
    
    models = {
        'ExponentialSmoothing': ExponentialSmoothing(),
        'ARIMA': ARIMA(),
        'Prophet': Prophet(),
        'LightGBM': LightGBMModel(lags=30),
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(train)
            pred = model.predict(horizon)
            error = mae(val, pred)
            results[name] = error
        except Exception as e:
            results[name] = float('inf')
            print(f"{name} failed: {e}")
    
    # Pick best model
    best_model = min(results, key=results.get)
    return best_model, results
```

---

## 10. Robust Cross-Validation

### 10.1 Expanding Window Cross-Validation [^1430^]

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def expanding_window_cv(data, model_class, min_train_size, test_size, horizon, model_kwargs=None):
    """
    Expanding window cross-validation that works for any time series.
    Mimics real-world deployment where data grows over time.
    """
    n = len(data)
    predictions = []
    actuals = []
    fold_metrics = []
    
    n_folds = (n - min_train_size) // test_size
    
    for i in range(n_folds):
        train_end = min_train_size + (i * test_size)
        test_start = train_end
        test_end = min(test_start + test_size, n)
        
        if test_end > n:
            break
        
        train_data = data.iloc[:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Fit model
        model_kwargs = model_kwargs or {}
        model = model_class(**model_kwargs)
        model.fit(train_data)
        
        # Predict
        forecast = model.predict(horizon)
        
        actuals.extend(test_data.values)
        predictions.extend(forecast[:len(test_data)])
        
        fold_mae = mean_absolute_error(test_data.values, forecast[:len(test_data)])
        fold_rmse = np.sqrt(mean_squared_error(test_data.values, forecast[:len(test_data)]))
        
        fold_metrics.append({
            'fold': i + 1,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'mae': fold_mae,
            'rmse': fold_rmse
        })
    
    overall_mae = mean_absolute_error(actuals, predictions)
    
    return {
        'overall_mae': overall_mae,
        'fold_metrics': fold_metrics,
        'n_folds': len(fold_metrics)
    }
```

### 10.2 Walk-Forward Validation (Most Robust) [^1429^]

> "Walk-forward optimization is arguably the most common and practical cross-validation method for time series, as it closely mimics how an algorithmic strategy is developed and deployed in a production environment." - [Algovantis](https://algovantis.com/cross-validation-methods-tailored-for-time-series-backtesting/)

```python
def walk_forward_cv(df, date_col, value_col, train_periods, test_periods, model_fn):
    """
    Walk-forward validation - the gold standard for time series.
    Training window expands, test window slides forward.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    results = []
    
    start = train_periods
    while start + test_periods <= len(df):
        train_df = df.iloc[:start]
        test_df = df.iloc[start:start + test_periods]
        
        # Train
        model = model_fn()
        model.fit(train_df)
        
        # Predict
        preds = model.predict(len(test_df))
        
        # Score
        actual = test_df[value_col].values
        mae = np.mean(np.abs(actual - preds))
        
        results.append({
            'train_end': start,
            'test_start': start,
            'test_end': start + test_periods,
            'mae': mae
        })
        
        start += test_periods
    
    return pd.DataFrame(results)
```

### 10.3 StatsForecast Built-in CV

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# StatsForecast has built-in robust CV
sf = StatsForecast(models=[AutoARIMA()], freq='D')

# Cross-validation: automatically handles temporal ordering
cv_df = sf.cross_validation(
    df=df,
    h=30,           # Forecast horizon
    step_size=30,   # Slide window by 30 days
    n_windows=3,    # 3 validation folds
)
```

---

## 11. Fallback Strategies

### 11.1 Naive Forecasting as Fallback [^1543^]

Microsoft Dynamics 365 explicitly defines naive forecasting as a **fallback for low-data scenarios** [^1543^]:

> "Naive forecasting isn't a forecasting model but a fallback method for low-data scenarios... Naive forecasting uses straightforward rules to produce approximate forecasts, reducing the uncertainty that arises from using regular models trained on limited datasets."

```python
class FallbackForecasters:
    """
    Collection of fallback forecasting methods for when:
    - Data is insufficient for complex models
    - Complex models fail during training
    - Data quality is poor
    """
    
    @staticmethod
    def naive(series, horizon):
        """Repeat the last observed value."""
        last_value = series.iloc[-1]
        return np.full(horizon, last_value)
    
    @staticmethod
    def seasonal_naive(series, horizon, season_length=7):
        """Repeat values from the same period in previous season(s)."""
        if len(series) < season_length:
            return FallbackForecasters.naive(series, horizon)
        
        # Take the last `season_length` values and repeat
        seasonal_pattern = series.iloc[-season_length:].values
        forecasts = []
        for i in range(horizon):
            forecasts.append(seasonal_pattern[i % season_length])
        return np.array(forecasts)
    
    @staticmethod
    def mean_forecast(series, horizon):
        """Use the historical mean."""
        return np.full(horizon, series.mean())
    
    @staticmethod
    def moving_average(series, horizon, window=30):
        """Use the recent moving average."""
        ma = series.iloc[-window:].mean()
        return np.full(horizon, ma)
    
    @staticmethod
    def linear_trend(series, horizon):
        """Extrapolate linear trend."""
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        future_x = np.arange(len(series), len(series) + horizon)
        return slope * future_x + intercept


def select_fallback(series, horizon, season_length=7):
    """
    Intelligently select the best fallback method based on data characteristics.
    """
    n = len(series.dropna())
    
    if n < 2:
        return FallbackForecasters.naive(series, horizon)
    elif n < season_length * 2:
        # Not enough for seasonal pattern - use naive or moving average
        return FallbackForecasters.moving_average(series, horizon, window=max(n//2, 2))
    else:
        # Check if series has strong seasonality
        seasonal_pattern = series.iloc[-season_length*2:-season_length].values
        next_season = series.iloc[-season_length:].values
        seasonal_corr = np.corrcoef(seasonal_pattern, next_season)[0,1] if len(seasonal_pattern) == len(next_season) else 0
        
        if seasonal_corr > 0.5:
            return FallbackForecasters.seasonal_naive(series, horizon, season_length)
        else:
            return FallbackForecasters.moving_average(series, horizon)
```

### 11.2 Fallback Decision Logic

```python
class RobustForecaster:
    """
    Production-grade forecaster with intelligent fallback handling.
    """
    
    MIN_OBSERVATIONS = 14  # Minimum observations for complex models
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.fallback_used = False
        self.fallback_method = None
    
    def fit(self, series):
        """Fit with automatic fallback."""
        n = len(series.dropna())
        
        if n < self.MIN_OBSERVATIONS:
            self.fallback_used = True
            self.fallback_method = 'fallback'
            return self  # No fitting needed
        
        try:
            # Try complex model
            self.model = AutoARIMA(season_length=7)
            self.model.fit(series)
            self.fallback_used = False
        except Exception as e:
            print(f"Complex model failed: {e}. Using fallback.")
            self.fallback_used = True
            self.fallback_method = 'fallback'
        
        return self
    
    def predict(self, series, horizon):
        """Predict with automatic fallback."""
        if self.fallback_used:
            return select_fallback(series, horizon)
        
        try:
            return self.model.predict(horizon)
        except Exception as e:
            print(f"Prediction failed: {e}. Using fallback.")
            return select_fallback(series, horizon)
```

### 11.3 Handling Model Drift [^1498^][^1552^]

When to use fallbacks according to industry best practices [^1498^]:

> **"Pause the model and use a fallback... If the model errors are risky or costly, we can't just let it go and tolerate unreliable predictions... You can often design a set of rules that will be less precise but more robust than a rogue model."**

```python
def should_use_fallback(series, model=None, predictions=None):
    """
    Decision function: should we use fallback instead of model predictions?
    """
    reasons = []
    
    # 1. Not enough data
    if len(series.dropna()) < 14:
        reasons.append("insufficient_data")
    
    # 2. All zeros or constant
    if series.nunique() <= 1:
        reasons.append("constant_series")
    
    # 3. Too many missing values
    if series.isna().mean() > 0.5:
        reasons.append("too_many_missing")
    
    # 4. Model predictions are negative (for demand data)
    if predictions is not None and np.any(predictions < 0):
        reasons.append("negative_predictions")
    
    # 5. Predictions wildly different from history
    if predictions is not None:
        historical_mean = series.mean()
        pred_mean = np.mean(predictions)
        if historical_mean > 0 and abs(pred_mean - historical_mean) / historical_mean > 10:
            reasons.append("unreasonable_predictions")
    
    return len(reasons) > 0, reasons
```

---

## 12. Production System Architecture

### 12.1 Reference Architecture [^1521^][^1523^][^1528^]

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                 │
│  CSV Files │ Excel │ Databases │ APIs │ Cloud Storage            │
└─────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Schema   │  │ Type     │  │ Format   │  │ Encoding │         │
│  │ Detector │  │ Inference│  │ Adapter  │  │ Handler  │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │ Data     │  │ Quality  │  │ Schema   │                       │
│  │ Type     │  │ Checks   │  │ Drift    │                       │
│  │ Checks   │  │          │  │ Detection│                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Missing  │  │ Frequency│  │ Resample │  │ Outlier  │         │
│  │ Value    │  │ Inference│  │ & Fill   │  │ Handling │         │
│  │ Fill     │  │          │  │ Gaps     │  │          │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Lag      │  │ Rolling  │  │ Date/    │  │ Holiday  │         │
│  │ Features │  │ Stats    │  │ Calendar │  │ Features │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                    MODELING LAYER                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Multiple │  │ Cross-   │  │ Best     │  │ Ensemble │         │
│  │ Models   │  │ Validat. │  │ Model    │  │ Optional │         │
│  │ (ARIMA,  │  │ (Expanding│  │ Select.  │  │          │         │
│  │  ETS, ML)│  │  Window) │  │          │  │          │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                    FALLBACK LAYER                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │ Naive    │  │ Seasonal │  │ Moving   │                       │
│  │          │  │ Naive    │  │ Average  │                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                  │
│  Forecasts │ Confidence Intervals │ Metadata │ Model Used        │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Pipeline Orchestration Options

| Tool | Best For | Key Feature |
|---|---|---|
| **Prefect** | Small teams, ML pipelines | Python-first, minimal boilerplate [^1521^] |
| **Dagster** | Data platforms, lineage tracking | Asset-oriented, strong typing [^1521^] |
| **Airflow** | Enterprise, large teams | Mature, 1000+ operators [^1523^] |

### 12.3 Key Production Patterns [^1528^]

- **Idempotency**: Re-runnable tasks with deterministic outputs
- **Write-Audit-Publish**: Validate data before production use
- **SLA Monitoring**: Alert on prediction latency, accuracy degradation
- **Exponential Backoff**: Retry failed model training
- **Sensors/Triggers**: Event-driven retraining on data changes

---

## 13. MLflow Integration & Model Registry

### 13.1 Model Registry for Forecasting [^1518^][^1520^][^1529^][^1530^]

```python
import mlflow
from mlflow.tracking import MlflowClient

def log_forecasting_model(model, model_name, params, metrics, signature=None, 
                          sample_input=None, sample_output=None):
    """
    Log a forecasting model to MLflow with full metadata.
    Enables model versioning, comparison, and rollback.
    """
    mlflow.set_experiment("demand_forecasting")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Infer signature if not provided
        if signature is None and sample_input is not None:
            signature = mlflow.models.infer_signature(sample_input, sample_output)
        
        # Log model with registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name
        )
        
        return mlflow.active_run().info.run_id


def load_production_model(model_name, alias="champion"):
    """
    Load the production model using alias pattern.
    Aliases are mutable pointers - update alias to deploy new model.
    """
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.pyfunc.load_model(model_uri)


class ModelVersionManager:
    """
    Manage model versions with champion/challenger pattern.
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = MlflowClient()
    
    def promote_to_champion(self, version):
        """Promote a model version to production champion."""
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="champion",
            version=version
        )
    
    def set_challenger(self, version):
        """Set a model version as challenger (for A/B testing)."""
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="challenger",
            version=version
        )
    
    def get_model_performance(self):
        """Get performance metrics for all model versions."""
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        performance = []
        for v in versions:
            run = self.client.get_run(v.run_id)
            performance.append({
                'version': v.version,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'stage': v.current_stage
            })
        return performance
```

### 13.2 Automated Model Promotion [^1518^]

```python
def auto_model_promotion(new_run_id, model_name, metric_name='mae', threshold_improvement=0.05):
    """
    Automatically promote a new model if it beats the champion.
    Returns: True if promoted, False otherwise.
    """
    client = MlflowClient()
    
    # Get new model metrics
    new_run = client.get_run(new_run_id)
    new_metric = new_run.data.metrics.get(metric_name, float('inf'))
    
    # Get champion metrics
    try:
        champion_versions = client.get_latest_versions(model_name, stages=["Production"])
        if champion_versions:
            champion_run = client.get_run(champion_versions[0].run_id)
            champion_metric = champion_run.data.metrics.get(metric_name, float('inf'))
            
            # Check if new model is better
            improvement = (champion_metric - new_metric) / champion_metric
            if improvement > threshold_improvement:
                # Promote!
                model_version = mlflow.register_model(
                    f"runs:/{new_run_id}/model", model_name
                )
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Production"
                )
                print(f"Promoted v{model_version.version}: {improvement:.1%} improvement")
                return True
    except Exception:
        pass
    
    return False
```

---

## 14. Complete Implementation Code

### 14.1 The Complete Generic Forecasting System

```python
#!/usr/bin/env python3
"""
Robust Generic Forecasting System
Works with ANY data format, ANY year, ANY schema.
"""

import pandas as pd
import numpy as np
import yaml
import holidays
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Statistical models
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoTheta, AutoCES, 
    Naive, SeasonalNaive, HistoricAverage
)
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class ForecastConfig:
    """Load and manage configuration from YAML or dict."""
    
    DEFAULT_CONFIG = {
        'data': {
            'date_column': None,
            'id_column': None, 
            'value_columns': None,
            'frequency': None,
            'input_format': 'auto',
        },
        'preprocessing': {
            'missing_value_strategy': 'auto',
            'regularize': True,
        },
        'features': {
            'auto_lags': True,
            'lag_periods': 'auto',
            'auto_rolling': True,
            'rolling_windows': 'auto',
            'date_features': True,
            'holidays': {'enabled': True, 'country': 'US', 'years': 'auto'},
        },
        'models': {
            'candidates': ['auto_arima', 'ets', 'theta', 'naive', 'seasonal_naive'],
            'cv_folds': 3,
            'metric': 'mae',
        },
        'fallback': {
            'enabled': True,
            'min_observations': 14,
            'method': 'auto',
        },
        'forecast': {
            'horizon': 30,
            'confidence_interval': 0.95,
        }
    }
    
    def __init__(self, config_path=None, config_dict=None):
        if config_path:
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
        elif config_dict:
            user_config = config_dict
        else:
            user_config = {}
        
        self.config = self._deep_merge(self.DEFAULT_CONFIG.copy(), user_config)
    
    def _deep_merge(self, d1, d2):
        """Recursively merge dictionaries."""
        for key, val in d2.items():
            if isinstance(val, dict) and key in d1 and isinstance(d1[key], dict):
                d1[key] = self._deep_merge(d1[key], val)
            else:
                d1[key] = val
        return d1
    
    def get(self, path, default=None):
        keys = path.split('.')
        val = self.config
        for key in keys:
            val = val.get(key, default) if isinstance(val, dict) else default
        return val


# ═══════════════════════════════════════════════════════════════
# SECTION 2: SCHEMA DETECTION
# ═══════════════════════════════════════════════════════════════

class SchemaDetector:
    """Automatically detect data schema - works with any column names."""
    
    DATE_KEYWORDS = ['date', 'time', 'day', 'ds', 'timestamp', 'period', 'fecha', 'datum']
    ID_KEYWORDS = ['id', 'key', 'code', 'sku', 'customer', 'product', 'series']
    
    def detect(self, df: pd.DataFrame, config: ForecastConfig) -> Dict:
        """Full schema detection pipeline."""
        result = {}
        
        # Detect date column
        result['date_column'] = config.get('data.date_column') or self._detect_date_column(df)
        
        # Detect ID column
        result['id_column'] = config.get('data.id_column') or self._detect_id_column(df, result['date_column'])
        
        # Detect value columns
        result['value_columns'] = config.get('data.value_columns') or \
                                   self._detect_value_columns(df, result['date_column'], result['id_column'])
        
        # Detect frequency
        result['frequency'] = config.get('data.frequency') or \
                              self._detect_frequency(df, result['date_column'], result['id_column'])
        
        result['date_range'] = (df[result['date_column']].min(), df[result['date_column']].max())
        result['n_series'] = df[result['id_column']].nunique() if result['id_column'] else 1
        result['n_rows'] = len(df)
        
        return result
    
    def _detect_date_column(self, df: pd.DataFrame) -> str:
        # Try column name matching
        for col in df.columns:
            if any(kw in str(col).lower() for kw in self.DATE_KEYWORDS):
                try:
                    pd.to_datetime(df[col], errors='raise')
                    return col
                except:
                    continue
        
        # Try type inference
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() / len(df) > 0.8:
                return col
        
        raise ValueError("Cannot auto-detect date column")
    
    def _detect_id_column(self, df: pd.DataFrame, date_col: str) -> Optional[str]:
        best_col, best_score = None, -1
        for col in df.columns:
            if col == date_col:
                continue
            n_unique = df[col].nunique()
            if 2 <= n_unique <= min(10000, int(len(df)**0.5)):
                score = 0
                if any(kw in str(col).lower() for kw in self.ID_KEYWORDS):
                    score += 100
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
                    score += 50
                score += min(n_unique, 500)
                if score > best_score:
                    best_score = score
                    best_col = col
        return best_col
    
    def _detect_value_columns(self, df: pd.DataFrame, date_col: str, id_col: Optional[str]) -> List[str]:
        exclude = {date_col, id_col}
        value_cols = []
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                value_cols.append(col)
        if not value_cols:
            raise ValueError("No numeric value columns found")
        return value_cols
    
    def _detect_frequency(self, df, date_col, id_col):
        sample = df[df[id_col] == df[id_col].iloc[0]] if id_col else df
        sample = sample.copy()
        sample[date_col] = pd.to_datetime(sample[date_col])
        sample = sample.sort_values(date_col)
        
        freq = pd.infer_freq(pd.DatetimeIndex(sample[date_col]))
        if freq:
            return freq
        
        # Fallback: median diff
        diffs = sample[date_col].diff().dropna()
        median = diffs.median()
        
        if median <= pd.Timedelta(hours=2): return 'H'
        elif median <= pd.Timedelta(days=2): return 'D'
        elif median <= pd.Timedelta(days=8): return 'W'
        elif median <= pd.Timedelta(days=35): return 'ME'
        return 'D'


# ═══════════════════════════════════════════════════════════════
# SECTION 3: DATA LOADER (MULTIPLE FORMATS)
# ═══════════════════════════════════════════════════════════════

class DataLoader:
    """Load data from any format: CSV, Excel, Parquet, JSON."""
    
    @staticmethod
    def load(filepath: str, format_hint: str = 'auto', **kwargs) -> pd.DataFrame:
        path = Path(filepath)
        
        if format_hint == 'auto':
            format_hint = path.suffix.lower()
        
        loaders = {
            '.csv': lambda: pd.read_csv(filepath, **kwargs),
            '.xlsx': lambda: pd.read_excel(filepath, **kwargs),
            '.xls': lambda: pd.read_excel(filepath, **kwargs),
            '.parquet': lambda: pd.read_parquet(filepath, **kwargs),
            '.json': lambda: pd.read_json(filepath, **kwargs),
        }
        
        if format_hint in loaders:
            df = loaders[format_hint]()
        elif format_hint == 'auto':
            # Try each loader
            for ext, loader in loaders.items():
                try:
                    df = loader()
                    break
                except:
                    continue
            else:
                raise ValueError(f"Cannot load file: {filepath}")
        else:
            raise ValueError(f"Unsupported format: {format_hint}")
        
        return df


# ═══════════════════════════════════════════════════════════════
# SECTION 4: PREPROCESSOR
# ═══════════════════════════════════════════════════════════════

class DataPreprocessor:
    """Clean and preprocess time series data automatically."""
    
    def preprocess(self, df: pd.DataFrame, schema: Dict, config: ForecastConfig) -> pd.DataFrame:
        df = df.copy()
        date_col = schema['date_column']
        id_col = schema['id_column']
        value_cols = schema['value_columns']
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Handle missing values
        strategy = config.get('preprocessing.missing_value_strategy', 'auto')
        for col in value_cols:
            if strategy == 'auto':
                df[col] = df.groupby(id_col)[col].transform(
                    lambda x: x.ffill().bfill()
                ) if id_col else df[col].ffill().bfill()
            elif strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'interpolate':
                df[col] = df[col].interpolate().bfill().ffill()
        
        # Regularize (fill date gaps)
        if config.get('preprocessing.regularize', True):
            df = self._regularize(df, schema)
        
        return df
    
    def _regularize(self, df, schema):
        date_col = schema['date_column']
        id_col = schema['id_column']
        value_cols = schema['value_columns']
        freq = schema['frequency']
        
        if id_col:
            result_dfs = []
            for group_id, group_df in df.groupby(id_col):
                group_df = group_df.sort_values(date_col).set_index(date_col)
                full_range = pd.date_range(group_df.index.min(), group_df.index.max(), freq=freq)
                group_df = group_df.reindex(full_range)
                group_df[date_col] = group_df.index
                group_df[id_col] = group_id
                result_dfs.append(group_df.reset_index(drop=True))
            df = pd.concat(result_dfs, ignore_index=True)
        else:
            df = df.sort_values(date_col).set_index(date_col)
            full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
            df = df.reindex(full_range)
            df[date_col] = df.index
            df = df.reset_index(drop=True)
        
        for col in value_cols:
            df[col] = df[col].ffill().bfill()
        
        return df


# ═══════════════════════════════════════════════════════════════
# SECTION 5: FEATURE ENGINEER
# ═══════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Automatically create time series features."""
    
    def engineer(self, df: pd.DataFrame, schema: Dict, config: ForecastConfig) -> pd.DataFrame:
        df = df.copy()
        date_col = schema['date_column']
        
        # Holiday features (GENERIC - works for any year)
        if config.get('features.holidays.enabled', True):
            df = self._add_holiday_features(df, date_col, config)
        
        # Date features
        if config.get('features.date_features', True):
            df = self._add_date_features(df, date_col)
        
        # Lag features
        if config.get('features.auto_lags', True):
            df = self._add_lag_features(df, schema, config)
        
        # Rolling features
        if config.get('features.auto_rolling', True):
            df = self._add_rolling_features(df, schema, config)
        
        return df
    
    def _add_holiday_features(self, df, date_col, config):
        country = config.get('features.holidays.country', 'US')
        years = config.get('features.holidays.years', 'auto')
        
        if years == 'auto':
            years = df[date_col].dt.year.unique().tolist()
            max_yr = max(years) if years else 2024
            years = list(range(min(years) if years else 2020, max_yr + 3))
        
        try:
            holiday_cal = holidays.country_holidays(country, years=years, observed=True)
            df['is_holiday'] = df[date_col].apply(lambda x: 1 if x.date() in holiday_cal else 0)
        except:
            df['is_holiday'] = 0
        
        df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        df['month'] = df[date_col].dt.month
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        
        return df
    
    def _add_date_features(self, df, date_col):
        dt = pd.to_datetime(df[date_col])
        df['year'] = dt.dt.year
        df['weekofyear'] = dt.dt.isocalendar().week.astype(int)
        df['is_month_start'] = dt.dt.is_month_start.astype(int)
        df['is_month_end'] = dt.dt.is_month_end.astype(int)
        df['days_since_start'] = (dt - dt.min()).dt.days
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        return df
    
    def _add_lag_features(self, df, schema, config):
        lags = config.get('features.lag_periods', 'auto')
        if lags == 'auto':
            lags = [1, 7, 14, 28]
        
        id_col = schema['id_column']
        value_cols = schema['value_columns']
        
        for col in value_cols:
            for lag in lags:
                if id_col:
                    df[f'{col}_lag_{lag}'] = df.groupby(id_col)[col].shift(lag)
                else:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df, schema, config):
        windows = config.get('features.rolling_windows', 'auto')
        if windows == 'auto':
            windows = [7, 14, 30]
        
        id_col = schema['id_column']
        value_cols = schema['value_columns']
        
        for col in value_cols:
            for window in windows:
                if id_col:
                    df[f'{col}_rolling_mean_{window}'] = df.groupby(id_col)[col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                    )
                else:
                    df[f'{col}_rolling_mean_{window}'] = df[col].shift(1).rolling(
                        window=window, min_periods=1).mean()
        
        return df


# ═══════════════════════════════════════════════════════════════
# SECTION 6: MODEL TRAINER WITH FALLBACK
# ═══════════════════════════════════════════════════════════════

class RobustModelTrainer:
    """Train models with automatic fallback for insufficient data."""
    
    MIN_OBSERVATIONS = 14
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.models = {}
        self.best_model_name = None
        self.used_fallback = {}
    
    def train(self, df: pd.DataFrame, schema: Dict) -> Dict:
        date_col = schema['date_column']
        id_col = schema['id_column']
        value_cols = schema['value_columns']
        freq = schema['frequency']
        results = {}
        
        # Determine season length
        season_map = {'D': 7, 'W': 52, 'ME': 12, 'YE': 1, 'H': 24}
        season_length = season_map.get(freq, 7)
        
        for value_col in value_cols:
            series_results = self._train_single_series(
                df, date_col, id_col, value_col, freq, season_length
            )
            results[value_col] = series_results
        
        return results
    
    def _train_single_series(self, df, date_col, id_col, value_col, freq, season_length):
        """Train on a single target column."""
        # Prepare data in StatsForecast format
        if id_col:
            sf_df = df[[id_col, date_col, value_col]].copy()
            sf_df.columns = ['unique_id', 'ds', 'y']
        else:
            sf_df = df[[date_col, value_col]].copy()
            sf_df.columns = ['ds', 'y']
            sf_df['unique_id'] = 'series_1'
        
        # Check if we need fallback
        series_counts = sf_df.groupby('unique_id').size()
        series_with_fallback = series_counts < self.MIN_OBSERVATIONS
        
        # Define models
        models = [
            AutoARIMA(season_length=season_length),
            AutoETS(season_length=season_length),
            AutoTheta(season_length=season_length),
            Naive(),
            SeasonalNaive(season_length=season_length),
        ]
        
        sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
        
        # Cross-validation
        try:
            cv_results = sf.cross_validation(
                df=sf_df, h=min(season_length * 2, 30),
                step_size=min(season_length, 7), n_windows=3
            )
            
            # Select best model
            eval_df = evaluate(cv_results, metrics=[mae])
            best_idx = eval_df['mae'].idxmin()
            best_model = eval_df.loc[best_idx, 'best_model']
            
            return {
                'trainer': sf,
                'best_model': best_model,
                'cv_results': cv_results,
                'evaluation': eval_df,
                'used_fallback': False,
            }
        except Exception as e:
            print(f"  CV failed, using fallback for {value_col}: {e}")
            return {
                'trainer': None,
                'best_model': 'fallback',
                'used_fallback': True,
                'fallback_reason': str(e),
            }
    
    def predict(self, df, schema, results, horizon):
        """Generate predictions with fallback handling."""
        predictions = {}
        
        for value_col, result in results.items():
            if result.get('used_fallback'):
                # Use fallback prediction
                pred = self._fallback_predict(df, schema, value_col, horizon)
            else:
                # Use model prediction
                try:
                    sf = result['trainer']
                    pred = sf.predict(h=horizon)
                except Exception as e:
                    print(f"  Model prediction failed, using fallback: {e}")
                    pred = self._fallback_predict(df, schema, value_col, horizon)
            
            predictions[value_col] = pred
        
        return predictions
    
    def _fallback_predict(self, df, schema, value_col, horizon):
        """Generate fallback predictions."""
        id_col = schema['id_column']
        date_col = schema['date_col'] if 'date_col' in schema else schema['date_column']
        freq = schema['frequency']
        
        if id_col:
            preds = []
            for group_id, group_df in df.groupby(id_col):
                series = group_df[value_col]
                last_date = group_df[date_col].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), 
                    periods=horizon, 
                    freq=freq
                )
                
                # Choose fallback method
                if len(series) >= 14:
                    values = FallbackForecasters.seasonal_naive(series, horizon, season_length=7)
                else:
                    values = FallbackForecasters.naive(series, horizon)
                
                pred_df = pd.DataFrame({
                    id_col: group_id,
                    'ds': future_dates,
                    'y': values
                })
                preds.append(pred_df)
            
            return pd.concat(preds, ignore_index=True)
        else:
            series = df[value_col]
            last_date = df[date_col].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=horizon, 
                freq=freq
            )
            values = FallbackForecasters.seasonal_naive(series, horizon, season_length=7)
            return pd.DataFrame({'ds': future_dates, 'y': values})


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

class GenericForecastingPipeline:
    """
    Complete generic forecasting pipeline.
    Works with ANY data, ANY year, ANY schema.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self.schema_detector = SchemaDetector()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = RobustModelTrainer(self.config)
        self.schema = None
        self.processed_df = None
        self.training_results = None
    
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from any file format."""
        format_hint = self.config.get('data.input_format', 'auto')
        return self.data_loader.load(filepath, format_hint, **kwargs)
    
    def detect_schema(self, df: pd.DataFrame) -> Dict:
        """Auto-detect data schema."""
        self.schema = self.schema_detector.detect(df, self.config)
        print(f"\n{'='*60}")
        print(f"DETECTED SCHEMA:")
        print(f"  Date column:    {self.schema['date_column']}")
        print(f"  ID column:      {self.schema['id_column']}")
        print(f"  Value columns:  {self.schema['value_columns']}")
        print(f"  Frequency:      {self.schema['frequency']}")
        print(f"  Date range:     {self.schema['date_range'][0]} to {self.schema['date_range'][1]}")
        print(f"  Series count:   {self.schema['n_series']}")
        print(f"  Row count:      {self.schema['n_rows']}")
        print(f"{'='*60}\n")
        return self.schema
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        self.processed_df = self.preprocessor.preprocess(df, self.schema, self.config)
        return self.processed_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create automatic features."""
        return self.feature_engineer.engineer(df, self.schema, self.config)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train models with auto-selection."""
        self.training_results = self.model_trainer.train(df, self.schema)
        return self.training_results
    
    def predict(self, horizon: int = None) -> Dict[str, pd.DataFrame]:
        """Generate forecasts."""
        if horizon is None:
            horizon = self.config.get('forecast.horizon', 30)
        
        predictions = self.model_trainer.predict(
            self.processed_df, self.schema, self.training_results, horizon
        )
        return predictions
    
    def run(self, filepath: str, horizon: int = None) -> Dict:
        """
        Run the complete pipeline in one call.
        
        Returns dict with:
            - schema: Detected schema
            - predictions: Forecast DataFrames per target column
            - metadata: Model info, quality metrics
        """
        # Step 1: Load
        print("Step 1: Loading data...")
        df = self.load_data(filepath)
        
        # Step 2: Detect schema
        print("Step 2: Detecting schema...")
        self.detect_schema(df)
        
        # Step 3: Preprocess
        print("Step 3: Preprocessing...")
        df = self.preprocess(df)
        
        # Step 4: Feature engineering
        print("Step 4: Engineering features...")
        df = self.engineer_features(df)
        
        # Step 5: Train
        print("Step 5: Training models...")
        results = self.train(df)
        
        # Step 6: Predict
        print("Step 6: Generating forecasts...")
        predictions = self.predict(horizon)
        
        # Compile metadata
        metadata = {
            'schema': self.schema,
            'model_results': {
                col: {
                    'best_model': res.get('best_model'),
                    'used_fallback': res.get('used_fallback', False),
                }
                for col, res in results.items()
            },
            'config': self.config.config,
        }
        
        return {
            'schema': self.schema,
            'predictions': predictions,
            'metadata': metadata,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 8: USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example 1: Minimal usage - just give it a file
    pipeline = GenericForecastingPipeline()
    
    # result = pipeline.run("/path/to/any_sales_data.csv", horizon=30)
    # for col, pred_df in result['predictions'].items():
    #     print(f"\nPredictions for {col}:")
    #     print(pred_df.head())
    
    # Example 2: With custom config
    custom_config = ForecastConfig(config_dict={
        'data': {'country': 'UK'},  # UK holidays
        'models': {'cv_folds': 5},
    })
    pipeline_uk = GenericForecastingPipeline(config=custom_config)
    
    # Example 3: Override auto-detected columns
    explicit_config = ForecastConfig(config_dict={
        'data': {
            'date_column': 'transaction_date',
            'id_column': 'customer_id',
            'value_columns': ['revenue'],
        }
    })
    pipeline_explicit = GenericForecastingPipeline(config=explicit_config)
```

### 14.2 Minimal Usage Example

```python
# ═══════════════════════════════════════════════════════════════
# MINIMAL USAGE - Just give it a file, it figures out the rest
# ═══════════════════════════════════════════════════════════════

from generic_forecasting import GenericForecastingPipeline

# Create pipeline (uses all defaults)
pipeline = GenericForecastingPipeline()

# Run on any data file
result = pipeline.run("my_sales_data.csv", horizon=30)

# Access predictions
for target_col, pred_df in result['predictions'].items():
    print(f"\n{target_col} forecast:")
    print(pred_df)

# Check what was auto-detected
print(f"Detected date column: {result['schema']['date_column']}")
print(f"Detected ID column: {result['schema']['id_column']}")
print(f"Detected frequency: {result['schema']['frequency']}")
```

---

## 15. Key Recommendations

### 15.1 Top 10 Implementation Rules

| # | Rule | Why |
|---|---|---|
| 1 | **Never hardcode column names** | Use SchemaDetector instead |
| 2 | **Never hardcode years** | Use `holidays` library with dynamic year detection |
| 3 | **Never assume data frequency** | Use `pd.infer_freq()` + fallback heuristics |
| 4 | **Always have a fallback** | Naive/seasonal naive for insufficient data |
| 5 | **Always validate input data** | Pandera checks prevent silent failures |
| 6 | **Use configuration files** | YAML configs for different environments |
| 7 | **Try multiple models** | Auto-select best via cross-validation |
| 8 | **Handle missing values automatically** | Forward-fill + backward-fill as default |
| 9 | **Support multiple file formats** | CSV, Excel, Parquet, JSON |
| 10 | **Log everything** | MLflow for model tracking, metrics, reproducibility |

### 15.2 How This System Handles Each Failure Mode

| Failure Mode | System Response |
|---|---|
| Different column names | SchemaDetector auto-detects date/value/ID columns |
| Different number of products | Dynamic groupby, handles any cardinality |
| Different date ranges | Auto-detects min/max from data |
| Different year (not 2026) | `holidays` library generates holidays for any year |
| Missing values | Auto-fill with forward-fill + backward-fill |
| Different file format | DataLoader supports CSV, Excel, Parquet, JSON |
| Insufficient data (<14 obs) | Falls back to naive/seasonal naive forecast |
| All zeros | Falls back to zero forecast |
| Negative values detected | Warns but handles; can clip to zero |
| Model training fails | Catches exception, uses fallback |
| Schema drift in production | DataQualityChecker flags changes |

### 15.3 Key Libraries Summary

| Library | Purpose | Why Use It |
|---|---|---|
| **StatsForecast** | Auto ARIMA/ETS/Theta | 20x faster than pmdarima [^1477^] |
| **Darts** | Unified model API | Same fit()/predict() for ALL models [^1538^] |
| **Prophet** | Seasonality + holidays | Handles missing data, outliers [^1534^] |
| **Pandas** | Data manipulation | `infer_freq()`, resample, reindex |
| **Holidays** | Holiday calendars | 100+ countries, any year [^1542^] |
| **Pandera** | Data validation | Schema-as-code, type checking [^1539^] |
| **MLflow** | Model registry | Versioning, tracking, deployment [^1529^] |
| **TSFresh** | Feature extraction | 100s of automatic features [^1479^] |
| **MLForecast** | ML-based forecasting | sklearn integration, lag transforms [^70^] |

### 15.4 Critical Design Decisions

1. **Configuration over code**: Every tunable parameter lives in YAML, not Python
2. **Fail gracefully**: Every component has a fallback to simpler methods
3. **Auto-detect everything**: Schema, frequency, seasonality - never assume
4. **Multi-model ensemble**: Try several, pick best via CV
5. **Temporal validation ONLY**: Never use random train/test split for time series
6. **Holiday-aware**: Use dynamic holiday lookups, never hardcode dates
7. **Monitor and adapt**: Track drift, auto-retrain when needed [^1552^]

---

## References

- [Nixtla StatsForecast - Automatic Model Selection](https://www.nixtla.io/blog/statsforecast-automatic-model-selection)
- [Nixtla StatsForecast - Automatic Forecasting](https://nixtlaverse.nixtla.io/statsforecast/docs/how-to-guides/automatic_forecasting.html)
- [Nixtla StatsForecast GitHub](https://github.com/nixtla/statsforecast)
- [Darts - Time Series Made Easy](https://github.com/unit8co/darts)
- [AutoForecast Paper (CIKM'22)](http://ryanrossi.com/pubs/AutoForecast-CIKM22.pdf)
- [VEST: Automatic Feature Engineering for Forecasting](https://link.springer.com/article/10.1007/s10994-021-05959-y)
- [Pandera Data Validation](https://endjin.com/blog/2023/03/a-look-into-pandera-and-great-expectations-for-data-validation)
- [Python Holidays Library](https://www.geeksforgeeks.org/python/python-holidays-library/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Evidently AI - Drift Detection](https://www.evidentlyai.com/)
- [Microsoft - Naive Forecasting Fallback](https://learn.microsoft.com/en-us/dynamics365/supply-chain/demand-planning/naive-forecast-algorithm)
- [Walk-Forward CV - Algovantis](https://algovantis.com/cross-validation-methods-tailored-for-time-series-backtesting/)
- [TSFresh Feature Extraction](https://github.com/blue-yonder/tsfresh)
- [Prefect vs Dagster Comparison](https://www.getorchestra.io/guides/prefect-vs-dagster-key-differences-2024)

---

*Document generated from 20+ web searches across academic papers, library documentation, industry blogs, and GitHub repositories. All code examples are original implementations synthesized from research findings.*
