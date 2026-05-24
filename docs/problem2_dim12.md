# Dimension 12: Validation Strategies for Extremely Short Time Series + Free External Data Sources

## Research Summary

This document covers validation strategies critical for extremely short time series (2 months / ~60 days of daily data) and practical free external data sources for enriching China-based demand forecasting models. Both sections focus on **implementable, student-team-friendly** solutions.

---

## PART A: VALIDATION STRATEGIES

### 1. Key Findings

#### Finding 1: Standard k-fold CV is fundamentally broken for time series

```
Claim: Standard k-fold cross-validation is invalid for time series because it violates temporal ordering and creates data leakage by training on future data to predict the past. [^1^]
Source: Cross-Validation Pitfalls - MetricGate
URL: https://metricgate.com/blogs/cross-validation-pitfalls/
Date: 2025
Excerpt: "A random 80/20 split on time-series data is not validation — it is cheating. The future must always be the test set."
Confidence: high
```

#### Finding 2: For 2 months (~60 days) of data, use 3-4 folds max with TimeSeriesSplit

```
Claim: With ~60 days of data, the maximum practical folds is 3-4. With n_splits=3 and sklearn's TimeSeriesSplit, you get training sets of ~15, ~30, ~45 days with test sets of ~15 days each. The gap parameter should be set to 1-2 to prevent leakage. [^2^]
Source: sklearn.model_selection.TimeSeriesSplit — scikit-learn 1.8.0 documentation
URL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
Date: 2024 (stable)
Excerpt: "Time Series cross-validator... In the k-th split, it returns the first k folds as the train set and the (k+1)-th fold as the test set."
Confidence: high
```

#### Finding 3: The M5 winner used last four 28-day windows for validation

```
Claim: The M5 forecasting competition winner (1st place) used the last four 28-day-long windows of available data for cross-validation, measuring both mean and standard deviation of errors to select models that were both accurate and robust. [^3^]
Source: The M5 Accuracy competition: Results, findings and conclusions
URL: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
Date: 2021
Excerpt: "The method was fine-tuned using the last four 28-day-long windows of available data for CV and by measuring both the mean and the standard deviation of the errors produced by the individual models and their combinations."
Confidence: high
```

#### Finding 4: M5 4th place used 5 holdout periods instead of rolling CV

```
Claim: The 4th place solution in M5 used 5 specific holdout periods (d1578-d1605, d1830-d1857, d1858-d1885, d1886-d1913, d1914-d1941) representing different historical regimes rather than consecutive rolling windows, demonstrating that diverse validation periods can be more valuable than many folds. [^4^]
Source: M5 Forecasting - 4th place solution (Kaggle)
URL: https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/monsaraida-4th-place-solution
Date: 2020
Excerpt: "5 holdout (d1578-d1605, d1830-d1857, d1858-d1885, d1886-d1913, d1914-d1941)... no early stopping"
Confidence: high
```

#### Finding 5: For extremely short series, prefer single holdout over many small folds

```
Claim: With only 60 days of data, a single holdout of the last 14 days is often more reliable than multiple tiny folds with insufficient training data. The holdout should be at least as long as your forecast horizon. [^5^]
Source: What about the training/test sets? - OpenForecast
URL: https://openforecast.org/2024/10/02/what-about-the-training-sets/
Date: 2024-10-02
Excerpt: "If your forecast horizon is 14 days ahead, your test set should have at least 14 observations for daily data. However, if you use exactly 14 observations, you'll only be able to do a fixed origin evaluation... A better approach is to make the test set longer than the forecast horizon."
Confidence: high
```

#### Finding 6: Purged cross-validation prevents leakage in time series with derived features

```
Claim: Purged K-Fold CV removes overlapping samples between training and test sets to prevent data leakage. The embargo method adds a temporal buffer around folds to further block future information contamination. This is critical when using rolling statistics features. [^6^]
Source: A Bayesian-based classification framework for financial time series trend prediction (Springer)
URL: https://link.springer.com/content/pdf/10.1007/s11227-022-04834-4.pdf
Date: 2022
Excerpt: "The solution for the second problem is to purge all overlapping labeled samples from the training and test sets. For the serial correlation problem between financial features, the solution is to embargo those samples in the series that immediately follow another sample in the test set."
Confidence: high
```

#### Finding 7: Group-based validation (leave-one-customer-out) is essential for hierarchical data

```
Claim: sklearn provides LeaveOneGroupOut for cross-validation where each split holds out one entire group. This prevents leakage across groups and is critical when the same time series patterns may not transfer across customers or warehouses. [^7^]
Source: scikit-learn LeaveOneGroupOut documentation
URL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
Date: 2024 (stable)
Excerpt: "Leave One Group Out cross-validator. Provides train/test indices to split data such that each training set is comprised of all samples except ones belonging to one specific group."
Confidence: high
```

#### Finding 8: Feature stability across validation splits is more important than single-fold performance

```
Claim: When evaluating models with limited data, feature importance stability across validation folds provides additional robustness evidence. Features that consistently rank highest across folds are more reliable than those that only perform well on one split. [^8^]
Source: Amberdata Research Paper
URL: https://go.amberdata.io/hubfs/AmberdataDefiStablecoinImpactPt5.pdf
Date: 2024
Excerpt: "Feature importance stability across validation folds provides additional robustness evidence. Repayment metrics consistently rank highest, confirming deleveraging as a universal volatility predictor transcending market conditions."
Confidence: high
```

---

### 2. Major Techniques

#### Technique 1: TimeSeriesSplit with sklearn (Primary Recommendation)

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# For ~60 days of data, use n_splits=3 with gap=2 to prevent leakage
# This gives you:
# Fold 1: Train days 0-15, Test days 18-33 (gap=2)
# Fold 2: Train days 0-30, Test days 33-48 (gap=2)
# Fold 3: Train days 0-45, Test days 48-60 (gap=2)

tscv = TimeSeriesSplit(
    n_splits=3,
    gap=2,              # Exclude 2 days between train and test to prevent leakage
    test_size=14        # 14-day test window
)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}: train={len(train_idx)}, test={len(test_idx)}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  Score: {score:.4f}")
```

#### Technique 2: Custom Expanding Window CV for Very Short Series

```python
import numpy as np
from typing import Generator, Tuple

def short_series_cv(
    n_samples: int,
    min_train_size: int = 21,   # At least 3 weeks
    test_size: int = 7,          # 1-week forecast horizon
    gap: int = 2,                # 2-day embargo
    n_splits: int = 3
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Custom expanding window CV optimized for very short time series.
    With n_samples=60, min_train_size=21, test_size=7, gap=2, n_splits=3:
    - Fold 1: train[0:21], test[23:30]
    - Fold 2: train[0:35], test[37:44]
    - Fold 3: train[0:49], test[51:58]
    """
    indices = np.arange(n_samples)

    for i in range(n_splits):
        train_end = min_train_size + i * ((n_samples - min_train_size - test_size - gap) // n_splits)
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)

        if test_end - test_start < test_size // 2:
            continue

        train_idx = indices[:train_end]
        test_idx = indices[test_start:test_end]

        yield train_idx, test_idx
```

#### Technique 3: mlforecast Cross-Validation (Production-Ready)

```python
from mlforecast import MLForecast
from sklearn.linear_model import Ridge
import lightgbm as lgb

fcst = MLForecast(
    models=[
        lgb.LGBMRegressor(verbosity=-1, n_estimators=100),
        Ridge(alpha=1.0),
    ],
    freq="D",
    lags=[1, 7, 14],
    lag_transforms={
        7: [RollingMean(window_size=7), RollingMean(window_size=14)],
    },
    date_features=["dayofweek", "month", "year"],
)

# For 60 days of data with 7-day horizon:
# n_windows=3 gives 3 validation periods
# h=7 predicts 7 days ahead
# step_size=7 moves forward 1 week at a time

cv_results = fcst.cross_validation(
    df=df,
    n_windows=3,
    h=7,
    step_size=7,
)

# Evaluate
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae

cv_metrics = evaluate(
    cv_results.drop(columns="cutoff"),
    metrics=[rmse, mae],
    agg_fn="mean",
)
print(cv_metrics)
```

#### Technique 4: Purged Cross-Validation (For Features with Lookback)

```python
class PurgedTimeSeriesSplit:
    """
    Time series CV with purging (gap between train and test) to prevent
    leakage when features have lookback windows (rolling means, lags).
    Based on Lopez de Prado's "Advances in Financial Machine Learning".
    """
    def __init__(self, n_splits=3, purge_gap=2, embargo_pct=0.01):
        self.n_splits = n_splits
        self.purge_gap = purge_gap  # Days to exclude between train/test
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold boundaries
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            # Purge: exclude gap days before test
            train_end = test_start - self.purge_gap
            if train_end <= fold_size // 2:
                continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
```

#### Technique 5: Group-Based Leave-One-Out (For Multiple Customers/Warehouses)

```python
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

# If you have multiple customers or warehouses, use group-based CV
# This ensures model generalizes to unseen customers/warehouses

logo = LeaveOneGroupOut()
groups = df['customer_id']  # or 'warehouse_id'

scores = cross_val_score(
    model, X, y,
    cv=logo,
    groups=groups,
    scoring='neg_mean_absolute_error'
)

print(f"Scores across groups: {scores}")
print(f"Mean: {-scores.mean():.4f}, Std: {scores.std():.4f}")
```

---

### 3. Implementation Details

#### Configuration for 60 Days of Data

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_splits | 3 | Maximum for 60 days; more folds = tiny train sets |
| test_size | 7-14 | Match forecast horizon (7 days) or double it (14) |
| gap | 1-2 | Prevent leakage from lag features |
| min_train_size | 21 | At least 3 weeks to learn weekly seasonality |
| max_train_size | None | Use expanding window (all available history) |

#### Example: Complete CV Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

# Sample: 60 days of data
dates = pd.date_range('2024-01-01', periods=60, freq='D')
X = pd.DataFrame({
    'lag_1': np.random.randn(60),
    'lag_7': np.random.randn(60),
    'rolling_mean_7': np.random.randn(60),
    'dayofweek': pd.Series(dates).dt.dayofweek.values,
})
y = np.random.randn(60)

# Recommended: TimeSeriesSplit with 3 folds, 2-day gap
cv = TimeSeriesSplit(n_splits=3, gap=2)

model = RandomForestRegressor(n_estimators=50, random_state=42)

# Custom scorer
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

scores = cross_val_score(model, X, y, cv=cv, scoring=mae_scorer)
print(f"MAE per fold: {-scores}")
print(f"Mean MAE: {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

### 4. What Works

1. **Expanding window CV**: Each fold uses all available history up to that point. This mimics real deployment where the model retrains on all past data. [^9^]
2. **Gap parameter (embargo)**: Setting gap=1-2 days between train and test prevents leakage from lag/rolling features. Essential when using lag_1, rolling_mean_7, etc.
3. **Matching test size to horizon**: Test set size should equal your forecast horizon (7 days for weekly forecasts, 14 for bi-weekly). [^5^]
4. **Multiple diverse holdout periods**: Instead of consecutive folds, use diverse periods (M5 4th place used 5 non-consecutive holdouts spanning different historical regimes). [^4^]
5. **Group-based CV**: When predicting for new customers/warehouses, test generalization with leave-one-group-out. [^7^]
6. **Stability-based model selection**: Choose models with low score variance across folds, not just lowest mean score. The M5 winner explicitly selected for both accuracy AND stability. [^3^]

---

### 5. What Doesn't Work

1. **Standard k-fold with shuffling**: Creates impossible scenarios (training on future data). Optimistic and misleading scores. [^1^]
2. **Too many folds with short data**: 5+ folds with 60 days gives training sets of <10 days - insufficient for any pattern learning.
3. **No gap with lag features**: Without a gap between train and test, lag_1 feature leaks the target directly.
4. **Fixed-size rolling window with short data**: A fixed 30-day window with 60 days gives only 1-2 folds and discards early data.
5. **Validation set too small**: Test sets of 1-2 days give high-variance, unreliable scores.
6. **Stratified sampling on time series**: Stratification assumes exchangeability which time series violate.

---

### 6. Competition Applications

```
Claim: In the M5 Accuracy competition, the top four performing methods and the vast majority of the top 50 submissions considered a CV strategy where at least the last four 28-day-long windows of available data were used to assess forecasting accuracy. The winner additionally measured both mean and standard deviation to select models that were both accurate and stable. [^3^]
Source: M5 Accuracy Competition paper
URL: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
Date: 2021
Excerpt: "The top four performing methods and the vast majority of the top 50 submissions considered a CV strategy where at least the last four 28-day-long windows of available data were used..."
Confidence: high
```

```
Claim: The M5 4th place solution explicitly abandoned complex CV and used simple holdouts: "I realized that the validation scores varied significantly over the period of time. I can't build a proper validation." Their practical 5-holdout approach still achieved 4th place. [^4^]
Source: M5 4th place Kaggle writeup
URL: https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/monsaraida-4th-place-solution
Date: 2020
Excerpt: "Then I realized that the validation scores varied significantly over the period of time. (I can't build a proper validation.)"
Confidence: high
```

---

### 7. Recommended Approach for Our Problem

Given **2 months (~60 days) of daily data**, here's our recommended validation strategy:

**Primary: Holdout + Light CV Hybrid**
1. **Reserve last 14 days** as final test set (never touch during development)
2. **Use days 32-46 as validation** (a middle period to test stability)
3. **Train on days 0-31** for initial model development
4. **For hyperparameter tuning**: Use TimeSeriesSplit with n_splits=3, gap=2 on the training data only
5. **Final evaluation**: Train on days 0-46, evaluate on days 47-60

**Rationale**: With only 60 days, a single reliable holdout > many tiny folds. The 14-day test matches a 2-week forecast horizon. Training on 45+ days gives enough data for weekly seasonality.

**Model Selection Criterion**: Choose the model with the best **worst-fold score**, not the best average. This ensures robustness across different time periods.

---

## PART B: FREE EXTERNAL DATA SOURCES

### 1. China Holidays API

#### Option A: Nager.Date (RECOMMENDED - Free, No Key, 100+ Countries)

```
Claim: Nager.Date is an open-source holiday Web API providing public holiday data for 100+ countries including China. No API key required, no rate limits, covers years 1925-2075. [^10^]
Source: Nager.Date API documentation
URL: https://date.nager.at
Date: 2024
Excerpt: "Free API WorldWide Public Holiday... No Rate Limits & CORS Support"
Confidence: high
```

**China endpoint**: `https://date.nager.at/api/v3/PublicHolidays/2024/CN`

```python
import requests

def get_china_holidays(year=2024):
    """Fetch China public holidays from Nager.Date (FREE, no API key)"""
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/CN"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    holidays = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(holidays)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'localName', 'name', 'fixed', 'global', 'types']]

# Usage
holidays_df = get_china_holidays(2024)
print(holidays_df)
```

**Sample response:**
```json
[
  {"date":"2024-01-01","localName":"元旦","name":"New Year's Day","countryCode":"CN","fixed":false,"global":true,"counties":null,"launchYear":null,"types":["Public"]},
  {"date":"2024-02-10","localName":"春节","name":"Spring Festival","countryCode":"CN","fixed":false,"global":true,"counties":null,"launchYear":null,"types":["Public"]},
  ...
]
```

**Features:**
- Returns Chinese names (localName) + English names
- Includes holiday types: Public, Bank, School, Optional, Observance
- Supports long weekend calculation endpoint
- **No API key, no rate limits, completely free**

#### Option B: jiejiariapi.com (Chinese-focused)

```
Claim: jiejiariapi.com provides a free China-specific holiday API with 50 requests/day for anonymous users, 100/day with API key. Returns holiday, weekend, and workday data. [^11^]
Source: jiejiariapi.com official site
URL: https://www.jiejiariapi.com/
Date: 2025
Excerpt: "Free: 100/day with API key, 50/day anonymous... Personal testing only; no commercial use"
Confidence: high
```

**Endpoints:**
- Holidays: `https://api.jiejiariapi.com/v1/holidays/2024`
- Weekends: `https://api.jiejiariapi.com/v1/weekends/2024`
- Workdays: `https://api.jiejiariapi.com/v1/workdays/2024`
- Is holiday: `https://api.jiejiariapi.com/v1/is_holiday?date=2024-01-01`

```python
import requests

def get_china_holidays_jiejiari(year=2024, api_key=None):
    """Fetch China holidays from jiejiariapi.com"""
    url = f"https://api.jiejiariapi.com/v1/holidays/{year}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()
```

#### Option C: Python workalendar/calendra Package (Offline)

```
Claim: The calendra Python package (fork of workalendar) includes China holiday calendars that are updated annually. Works offline once installed. [^12^]
Source: calendra documentation
URL: https://calendra.readthedocs.io/en/latest/history.html
Date: 2024
Excerpt: "Updated China for 2025... Added China holidays for 2024"
Confidence: high
```

```python
# pip install calendra
from calendra.asia import China

cal = China()
holidays = cal.holidays(2024)
# Returns list of (date, name) tuples
for date_obj, name in holidays:
    print(f"{date_obj}: {name}")
```

**Comparison Table:**

| Source | Cost | Key Required | Rate Limit | Chinese Names | English Names | Offline |
|--------|------|--------------|------------|---------------|---------------|---------|
| Nager.Date | Free | No | None | Yes | Yes | No |
| jiejiariapi | Free | Optional | 50-100/day | Yes | No | No |
| calendra | Free | N/A | N/A | Yes | No | Yes |

---

### 2. Weather APIs

#### Option A: Open-Meteo (RECOMMENDED - Best Free Option)

```
Claim: Open-Meteo is a completely free weather API with no API key required, 10,000 requests/day, historical data back to 1940, and includes built-in geocoding. Open-source and self-hostable. [^13^]
Source: Open-Meteo API documentation
URL: https://open-meteo.com/en/docs
Date: 2024
Excerpt: "Free access for non-commercial use. No API key required... Historical archive: Access over 80 years of hourly weather records"
Confidence: high
```

**Key capabilities:**
- **No API key required** - works immediately
- Historical data back to **1940**
- Daily variables: temperature_max, temperature_min, precipitation_sum, windspeed_max, etc.
- Built-in geocoding API (city name -> lat/lon)
- Python SDK: `pip install openmeteo-requests`

```python
import requests
import pandas as pd

def get_historical_weather(latitude, longitude, start_date, end_date):
    """
    Fetch historical weather from Open-Meteo (FREE, no API key)

    Parameters:
    -----------
    latitude : float  e.g., 31.2304 for Shanghai
    longitude : float  e.g., 121.4737 for Shanghai
    start_date : str  'YYYY-MM-DD'
    end_date : str  'YYYY-MM-DD'
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "windspeed_10m_max,relative_humidity_2m_mean",
        "timezone": "auto",
        "temperature_unit": "celsius",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Convert to DataFrame
    daily = data['daily']
    df = pd.DataFrame({
        'date': pd.to_datetime(daily['time']),
        'temp_max': daily['temperature_2m_max'],
        'temp_min': daily['temperature_2m_min'],
        'precipitation': daily['precipitation_sum'],
        'wind_max': daily['windspeed_10m_max'],
        'humidity_mean': daily['relative_humidity_2m_mean'],
    })
    return df

# Example: Get weather for Shanghai for competition period
shanghai_weather = get_historical_weather(
    latitude=31.2304,
    longitude=121.4737,
    start_date="2024-01-01",
    end_date="2024-02-29"
)
print(shanghai_weather.head())
```

**Built-in geocoding (no API key):**
```python
def get_coordinates(city):
    """Free geocoding via Open-Meteo"""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en"}
    response = requests.get(url, params=params)
    result = response.json()["results"][0]
    return result["latitude"], result["longitude"]

lat, lon = get_coordinates("Beijing")
# Returns: (39.9042, 116.4074)
```

#### Option B: OpenWeatherMap (Good Alternative)

```
Claim: OpenWeatherMap free tier provides 1,000 API calls/day with current weather, 5-day forecasts, and geocoding. Historical data requires paid tier. [^14^]
Source: OpenWeatherMap pricing
URL: https://openweathermap.org/price
Date: 2024
Excerpt: "Free tier: 1,000 API calls per day... Calls beyond free 1,000/day will be charged"
Confidence: high
```

```python
import requests

def get_weather_openweathermap(city, api_key):
    """OpenWeatherMap - requires free API key"""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    return response.json()
```

#### Option C: Visual Crossing Weather (1,000 records/day free)

```
Claim: Visual Crossing Weather API provides 50 years of historical data with 1,000 free records per day. Supports JSON and CSV. Good for bulk historical downloads. [^15^]
Source: Visual Crossing pricing page
URL: https://www.visualcrossing.com/weather-api/
Date: 2024
Excerpt: "Our weather data API allows 1000 completely free weather records per day"
Confidence: high
```

**Comparison Table:**

| API | Free Tier | Key Required | Historical Data | Rate Limit | Best For |
|-----|-----------|--------------|-----------------|------------|----------|
| **Open-Meteo** | 10K/day | **No** | **1940+** | ~10K/day | **Primary choice - no key needed** |
| OpenWeatherMap | 1K/day | Yes | Paid only | 60/min | Current weather, forecasts |
| Visual Crossing | 1K/day | Yes | 50+ years | 1K/day | Bulk historical data downloads |

---

### 3. City Data (Tier, GDP, Population)

#### Option A: simplemaps.com China Cities Database (Free, MIT License)

```
Claim: simplemaps provides a free subset of 367 prominent China cities with latitude, longitude, and other variables under MIT license. Available in CSV, JSON, and XLSX formats. [^16^]
Source: simplemaps China Cities Database
URL: https://simplemaps.com/data/cn-cities
Date: 2024
Excerpt: "Below is a list of 367 prominent cities in China. Each row includes a city's latitude, longitude, and other variables of interest... releasing this data subset for free under an MIT license"
Confidence: high
```

**Direct download:**
```python
import pandas as pd

# Download directly
url = "https://simplemaps.com/static/data/country-cities/cn/cn.csv"
cities_df = pd.read_csv(url)
print(cities_df.columns)
# ['city', 'city_ascii', 'lat', 'lng', 'country', 'iso2', 'iso3',
#  'admin_name', 'capital', 'population', 'id']

# Example: Get coordinates for a city
city_info = cities_df[cities_df['city'] == 'Shanghai'].iloc[0]
print(f"Shanghai: lat={city_info['lat']}, lon={city_info['lng']}, "
      f"pop={city_info['population']}")
```

#### China City Tier Classification (Manual)

```
Claim: China's city tier system is well-documented. Tier 1: Beijing, Shanghai, Guangzhou, Shenzhen. New Tier 1 includes Chengdu, Hangzhou, Chongqing, Suzhou, Wuhan, Xi'an, Nanjing, Changsha, Tianjin, Zhengzhou, Dongguan, Wuxi, Ningbo, Qingdao, Hefei. [^17^]
Source: Glopen - China's City Tiers Explained
URL: https://gl-open.com/blog/business/chinas-city-tiers-explained/
Date: 2025
Excerpt: "First-Tier (4 cities): Shanghai, Beijing, Shenzhen, Guangzhou... New First-Tier (15 cities): Chengdu, Hangzhou, Chongqing..."
Confidence: high
```

```python
# Create city tier mapping manually (authoritative from Yicai Global)
CITY_TIERS = {
    # Tier 1 (4 cities)
    'Shanghai': 1, 'Beijing': 1, 'Shenzhen': 1, 'Guangzhou': 1,
    # New Tier 1 (15 cities)
    'Chengdu': 1.5, 'Hangzhou': 1.5, 'Chongqing': 1.5, 'Suzhou': 1.5,
    'Wuhan': 1.5, "Xi'an": 1.5, 'Nanjing': 1.5, 'Changsha': 1.5,
    'Tianjin': 1.5, 'Zhengzhou': 1.5, 'Dongguan': 1.5, 'Wuxi': 1.5,
    'Ningbo': 1.5, 'Qingdao': 1.5, 'Hefei': 1.5,
    # Tier 2 (30 cities)
    'Foshan': 2, 'Shenyang': 2, 'Kunming': 2, 'Jinan': 2, 'Xiamen': 2,
    'Fuzhou': 2, 'Dalian': 2, 'Harbin': 2, 'Changchun': 2, 'Nanchang': 2,
    # ... add more as needed
}

def get_city_tier(city_name):
    return CITY_TIERS.get(city_name, 3)  # Default to tier 3
```

---

### 4. Geocoding (Converting Chinese Addresses to Lat/Lon)

#### Option A: Nominatim / OpenStreetMap (Free, Open Source)

```
Claim: Nominatim is a free geocoding service built on OpenStreetMap data. Supports Chinese queries with accept-language=zh parameter. Rate limited to 1 request/second for free public API. [^18^]
Source: Nominatim documentation / CSDN Chinese tutorial
URL: https://blog.csdn.net/Tomwildboar/article/details/139784584
Date: 2024
Excerpt: "accept-language=zh returns Chinese... zoom=8 returns city level"
Confidence: high
```

```python
import requests
import time

def geocode_chinese_address(address):
    """
    Geocode Chinese address using Nominatim (FREE)
    Rate limit: 1 request/second
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "addressdetails": 1,
        "accept-language": "zh-CN",
        "limit": 1,
        "countrycodes": "cn"  # Limit to China
    }
    headers = {
        "User-Agent": "SupplyChainForecast/1.0 (your-email@example.com)"
    }

    time.sleep(1)  # Respect rate limit (1 req/sec)
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    results = response.json()
    if results:
        return {
            'lat': float(results[0]['lat']),
            'lon': float(results[0]['lon']),
            'display_name': results[0]['display_name'],
            'type': results[0]['type']
        }
    return None

# Example: Geocode Chinese city
result = geocode_chinese_address("深圳市")
# Returns: {'lat': 22.5445741, 'lon': 114.0545429, ...}
```

#### Option B: Open-Meteo Geocoding API (Free, No Key)

```python
def geocode_city_op meteo(city_name):
    """
    Free geocoding via Open-Meteo (no API key, no rate limit)
    Best for major cities. Less detailed than Nominatim.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if "results" in data and data["results"]:
        result = data["results"][0]
        return {
            'lat': result['latitude'],
            'lon': result['longitude'],
            'name': result.get('name'),
            'country': result.get('country'),
            'admin1': result.get('admin1')  # Province
        }
    return None

# Works well for major Chinese cities
print(geocode_city_opmeteo("Shanghai"))
# {'lat': 31.2222, 'lon': 121.4581, 'name': 'Shanghai', 'country': 'China', 'admin1': 'Shanghai'}
```

#### Option C: geopy Python Package (Wraps Nominatim)

```python
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

def geocode_with_geopy(address):
    """
    Geocode using geopy + Nominatim (FREE)
    pip install geopy
    """
    geolocator = Nominatim(
        user_agent="supply-chain-forecast/1.0",
        timeout=10
    )

    try:
        time.sleep(1)  # Respect rate limits
        location = geolocator.geocode(address, language="zh")
        if location:
            return {
                'lat': location.latitude,
                'lon': location.longitude,
                'address': location.address
            }
    except GeocoderTimedOut:
        print(f"Timeout for address: {address}")

    return None
```

**Comparison:**

| Service | Cost | Key Required | Rate Limit | Chinese Support | Accuracy |
|---------|------|--------------|------------|-----------------|----------|
| Nominatim (OSM) | Free | No | 1 req/sec | Yes (zh param) | Good for cities |
| Open-Meteo Geo | Free | No | ~10K/day | Basic | Major cities only |
| geopy | Free | No | Same as Nominatim | Yes | Good |

---

### 5. Economic Indicators (China CPI, PPI, Retail Sales)

#### Option A: China National Bureau of Statistics (NBS) via stats.gov.cn

```
Claim: China's National Bureau of Statistics provides official CPI, PPI, retail sales, and other economic data through their English-language data portal. Registration required for bulk downloads. [^19^]
Source: China National Bureau of Statistics Database
URL: https://data.stats.gov.cn/staticreq.htm?m=aboutctryinfo
Date: 2024
Excerpt: "The National Bureau of Statistics database contains monthly, quarterly, annual data... Monthly data mainly include CPI, PPI..."
Confidence: high
```

**Access method:**
1. Visit: https://data.stats.gov.cn/easyquery.htm?cn=A01 (English version)
2. Browse to: Price -> Consumer Price Index
3. Download as CSV or copy from table

#### Option B: Direct Web Scraping (not recommended for competition)

For competition use, manual download from NBS is most reliable. Automated scraping may break.

#### Option C: pandas-datareader (Limited China Support)

```python
# pip install pandas-datareader
from pandas_datareader import data as pdr
import pandas as pd

# FRED has some China-related data but limited
# Example: China CPI (if available on FRED)
try:
    china_cpi = pdr.get_data_fred('CHNCPIALLMINMEI', start='2020-01-01')
    print(china_cpi.head())
except Exception as e:
    print(f"FRED China data may not be available: {e}")

# Note: pandas-datareader does NOT have comprehensive China economic data
# The library is primarily designed for US data (FRED, Yahoo Finance)
```

**Reality Check**: There is **no free, reliable, programmatic API** for China CPI/PPI/retail sales data. The official NBS website is the authoritative source but requires manual download or has an API that requires registration. For a student competition:

1. **Download manually from NBS** once at the start
2. **Include the CSV in your project repo**
3. **Merge with your training data** during preprocessing

---

### 6. Complete Integration Example

Here's a complete workflow combining all data sources:

```python
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# 1. GET CHINA HOLIDAYS (Nager.Date - free, no key)
# ============================================================
def fetch_china_holidays(year=2024):
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/CN"
    response = requests.get(url, timeout=10)
    holidays = response.json()
    return {pd.Timestamp(h['date']).strftime('%Y-%m-%d'): h['localName']
            for h in holidays}

# ============================================================
# 2. GET WEATHER DATA (Open-Meteo - free, no key)
# ============================================================
def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto"
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()['daily']
    return pd.DataFrame({
        'date': pd.to_datetime(data['time']),
        'temp_max': data['temperature_2m_max'],
        'temp_min': data['temperature_2m_min'],
        'precipitation': data['precipitation_sum']
    })

# ============================================================
# 3. GEOCODE CITIES (Open-Meteo geocoding - free, no key)
# ============================================================
def geocode_city(city_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1}
    response = requests.get(url, params=params)
    results = response.json().get('results', [])
    if results:
        return results[0]['latitude'], results[0]['longitude']
    return None, None

# ============================================================
# 4. BUILD FEATURES
# ============================================================
def build_external_features(df, city_name, year=2024):
    """
    Merge external data sources into training DataFrame.
    df must have a 'date' column.
    """
    # Get holidays
    holidays = fetch_china_holidays(year)
    df['is_holiday'] = df['date'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)
    df['holiday_name'] = df['date'].dt.strftime('%Y-%m-%d').map(holidays).fillna('')

    # Get weather
    lat, lon = geocode_city(city_name)
    if lat and lon:
        start = df['date'].min().strftime('%Y-%m-%d')
        end = df['date'].max().strftime('%Y-%m-%d')
        weather = fetch_weather(lat, lon, start, end)
        df = df.merge(weather, on='date', how='left')

    # Day of week / month features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    return df

# Usage:
# df = build_external_features(df, city_name='Shanghai')
```

---

## Complete Sources List

### Validation Strategies

| # | Source | URL | Date |
|---|--------|-----|------|
| 1 | Cross-Validation Pitfalls - MetricGate | https://metricgate.com/blogs/cross-validation-pitfalls/ | 2025 |
| 2 | sklearn TimeSeriesSplit docs | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html | 2024 |
| 3 | M5 Accuracy Competition Paper | https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf | 2021 |
| 4 | M5 4th Place Kaggle Writeup | https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/monsaraida-4th-place-solution | 2020 |
| 5 | OpenForecast - Training/Test Sets | https://openforecast.org/2024/10/02/what-about-the-training-sets/ | 2024 |
| 6 | Springer - Purged CV Paper | https://link.springer.com/content/pdf/10.1007/s11227-022-04834-4.pdf | 2022 |
| 7 | sklearn LeaveOneGroupOut | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html | 2024 |
| 8 | Amberdata Feature Stability | https://go.amberdata.io/hubfs/AmberdataDefiStablecoinImpactPt5.pdf | 2024 |
| 9 | Forward Chaining CV Guide | https://www.onenoughtone.com/learn/time-series-cross-validation | 2026 |
| 10 | MLForecast Cross-Validation | https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation.html | 2024 |

### Free External Data Sources

| # | Source | URL | Date |
|---|--------|-----|------|
| 10 | Nager.Date Holiday API | https://date.nager.at | 2024 |
| 11 | jiejiariapi.com | https://www.jiejiariapi.com/ | 2025 |
| 12 | calendra (Python China holidays) | https://calendra.readthedocs.io/ | 2024 |
| 13 | Open-Meteo Weather API | https://open-meteo.com/en/docs | 2024 |
| 14 | OpenWeatherMap | https://openweathermap.org/price | 2024 |
| 15 | Visual Crossing Weather | https://www.visualcrossing.com/weather-api/ | 2024 |
| 16 | simplemaps China Cities | https://simplemaps.com/data/cn-cities | 2024 |
| 17 | China City Tiers (Glopen) | https://gl-open.com/blog/business/chinas-city-tiers-explained/ | 2025 |
| 18 | Nominatim Geocoding | https://nominatim.openstreetmap.org/ | 2024 |
| 19 | China NBS Data Portal | https://data.stats.gov.cn/ | 2024 |
| 20 | Open-Meteo Historical API Python Example | https://gitcode.csdn.net/69ce6c7e54b52172bc668461.html | 2026 |

---

## Quick Reference Card

### For Validation
```
Data: 60 days daily | Horizon: 7 days | Model: LightGBM

Recommended: Single holdout (14 days) + 3-fold TimeSeriesSplit on train
- Never shuffle time series data
- Always use gap=1-2 when using lag features
- Test size = forecast horizon
- Select model by worst-fold performance, not average
```

### For External Data
```
Holidays:  Nager.Date API (FREE, no key) -> date.nager.at/api/v3/PublicHolidays/2024/CN
Weather:   Open-Meteo API (FREE, no key) -> archive-api.open-meteo.com/v1/archive
Geocode:   Open-Meteo Geo (FREE, no key) -> geocoding-api.open-meteo.com/v1/search
            OR Nominatim (FREE, 1 req/sec) -> nominatim.openstreetmap.org
Cities:    simplemaps (FREE, CSV) -> simplemaps.com/data/cn-cities
Economics: NBS China (manual download) -> data.stats.gov.cn
```
