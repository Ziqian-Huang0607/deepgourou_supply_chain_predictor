# Research Task 7: Feature Engineering for Short Time Series (<100 Days)

## Executive Summary

With only ~90 days of data, traditional feature engineering approaches that rely on long historical windows become problematic. This report synthesizes findings from academic papers, Kaggle competition winner writeups, and industry best practices to identify the **most effective feature engineering approaches for short time series**. The key insight: **with <100 days, prioritize domain-informed features, cross-series information sharing, and careful regularization** over automated feature extraction tools that require more data to be effective.

---

## Table of Contents

1. [Feature Engineering for Short Time Series (<100 Days)](#1-feature-engineering-for-short-time-series)
2. [Cross-Learning & Multi-Task Learning](#2-cross-learning--multi-task-learning)
3. [Embedding Categorical Features for Time Series](#3-embedding-categorical-features)
4. [Calendar Feature Engineering](#4-calendar-feature-engineering)
5. [Lag Feature Selection](#5-lag-feature-selection)
6. [Target Encoding for Time Series Categoricals](#6-target-encoding-for-time-series)
7. [Automated Feature Extraction (TSFresh)](#7-automated-feature-extraction-tsfresh)
8. [Feature Engineering for Sparse Time Series](#8-feature-engineering-for-sparse-time-series)
9. [Rolling Window Best Practices](#9-rolling-window-best-practices)
10. [Entity Embeddings for Time Series](#10-entity-embeddings)

---

## 1. Feature Engineering for Short Time Series (<100 Days)

### Key Findings

**Core Challenge**: With <100 days of data, many standard feature engineering techniques become problematic:
- Long lag features consume too many data points (e.g., lag-28 loses 28 rows)
- Rolling windows with large windows are unreliable
- Seasonal decomposition requires at least 2 full seasonal cycles
- Automated tools like TSFresh may overfit

**Kaggle Evidence (Learnings from Kaggle's Forecasting Competitions, arXiv 2020)**:
- In the Corporacion Favorita competition, winners **dropped older observations** and used only very recent data (1, 3, or 5 months) despite having 55 months available
- The winner's key insight: "The winner only used very recent data in the models, electing to drop older observations based on validation dataset performance"
- **Rolling statistics calculated at different hierarchy levels** were the most impactful features
- For short forecast horizons, recent data matters more than long history

**Recommended Approach for <100 Days**:

| Feature Type | Recommendation | Max Lag/Window |
|---|---|---|
| Lag features | Use short lags only (1-7 days) | 7 days max |
| Rolling windows | Small windows (3-7 days) | 14 days max |
| Calendar features | **Essential** - captures seasonality without data cost | N/A |
| Cross-series features | **Critical** - pool across series | N/A |
| Decomposition features | Only if sufficient data | N/A |

**Cold-Start Considerations**:
- Research shows that local models (GBDT) perform better on shortest series with only a few weeks of history (Drietomsky, TU Wien 2025)
- For cold-start items, use hierarchical/global features: historical averages at parent category level, similar item patterns
- Use static features (product metadata, price, category) more heavily since temporal features are limited

### Implementation Priority for <100 Days

**Tier 1 (Must-Have)**:
1. Calendar features (day of week, month, holidays) - zero data cost
2. Short lag features (lag 1-7)
3. Small rolling windows (3, 7-day means/stds)
4. Categorical encodings using hierarchy

**Tier 2 (High Value)**:
5. Cross-series pooled statistics (same-category averages)
6. Expanding window features (cumulative mean, cumulative std)
7. Proximity to holidays/events

**Tier 3 (If Data Permits)**:
8. Seasonal decomposition features
9. Fourier terms for seasonality
10. TSFresh (limited subset)

---

## 2. Cross-Learning & Multi-Task Learning

### Key Findings

**Core Principle**: When individual time series are short, pool information across related series to learn shared patterns.

**Evidence from Literature**:
- Kaggle winners consistently used **global models** that cross-learn from multiple series
- Walmart winner: "The winner accomplished this using a truncated SVD for each category of time series, which was used to reconstruct the individual series" - denoising via cross-series patterns
- DeepAR (Amazon): "When your dataset contains hundreds of related time series, DeepAR outperforms standard ARIMA and ETS methods"
- The approach: learn seasonality, trend, and event effects globally, apply locally

**Multi-Task Learning Approaches**:

1. **Hierarchical/Global Features**:
   - Compute statistics at higher aggregation levels (category avg, store avg)
   - Use these as features for individual series
   - Kaggle winners calculated "statistics at various levels of the hierarchy"

2. **Low-Rank Embeddings**:
   - Truncated SVD on item-activity matrix to learn latent representations
   - Used by Walmart winner for denoising
   - Must recompute within each CV fold to prevent leakage

3. **Global Neural Networks**:
   - Single model trained on all series simultaneously
   - Learns shared seasonal/trend patterns across all series
   - Uses series identifiers as categorical features
   - TFT (Temporal Fusion Transformer) is designed for this

4. **Cross-Learning via Features**:
   - Rolling/grouped statistics by category: `mean(sales) by [store, weekday]`, `mean(sales) by [product, month]`
   - These features "pool information from similar time series" (Kaggle paper)

### Suitability for <100 Days: **HIGH**

Cross-learning is **the most important strategy** for short time series. By pooling statistics across products/customers/stores, you effectively multiply your available data.

### Implementation Complexity: **Medium**

```python
# Example: Hierarchical cross-learning features
def create_cross_learning_features(df, group_cols, target_col):
    """Create pooled statistics at different hierarchy levels."""
    features = {}
    
    # Group-level mean (e.g., category average sales)
    for level in group_cols:
        features[f'{level}_{target_col}_mean'] = (
            df.groupby(level)[target_col].transform('mean')
        )
        features[f'{level}_{target_col}_std'] = (
            df.groupby(level)[target_col].transform('std')
        )
    
    # Interaction-level means (e.g., store x dayofweek)
    if len(group_cols) >= 2:
        features[f'{"_".join(group_cols[:2])}_{target_col}_mean'] = (
            df.groupby(group_cols[:2])[target_col].transform('mean')
        )
    
    return pd.DataFrame(features)
```

---

## 3. Embedding Categorical Features for Time Series

### Key Findings

**The Problem**: Customer IDs, Product IDs, Store IDs are high-cardinality categorical features that contain valuable information but cannot be one-hot encoded efficiently.

**Key Insight from Literature**:
- "Low-cardinality categorical variables (e.g., day of week, warehouse type) are one-hot encoded, while high-cardinality attributes (e.g., product IDs, customer IDs) are represented through trainable entity embeddings" (Multi-dataset LSTM analysis, 2026)
- DeepAR: "Embeddings of categorical features... transform categorical variables into continuous vectors. The inclusion of such features enhances the model's ability to discern patterns"
- TFT uses "categorical embeddings where a vector representation of the categories is learned"

**Encoding Strategies by Cardinality**:

| Cardinality | Method | Why |
|---|---|---|
| Low (<10) | One-hot encoding | No information loss, interpretable |
| Medium (10-100) | Target encoding with smoothing | Captures target relationship |
| High (>100) | Entity embeddings | Handles sparsity, learns similarity |

**For Short Time Series (<100 days)**:
- Entity embeddings are **especially valuable** because they learn from all series simultaneously
- A product with only 30 days of history can leverage learned embeddings from products with 90 days
- For tree-based models (XGBoost/LightGBM): use target encoding with time-aware cross-validation
- For neural networks: use embedding layers trained end-to-end

### Implementation Complexity: **Medium-High**

```python
# PyTorch: Entity embedding for categorical features
class EntityEmbedding(nn.Module):
    def __init__(self, n_categories, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(n_categories, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

# Tree-based: Target encoding with time-aware CV
from category_encoders import TargetEncoder

def temporal_target_encode(df, col, target, date_col, n_splits=5):
    """Target encoding that respects temporal order."""
    df = df.sort_values(date_col)
    encoder = TargetEncoder(cols=[col], smoothing=10)
    
    # Expanding window encoding
    encoded = np.zeros(len(df))
    fold_size = len(df) // n_splits
    
    for i in range(1, n_splits):
        train_idx = slice(0, i * fold_size)
        test_idx = slice(i * fold_size, (i+1) * fold_size)
        
        encoder.fit(df.iloc[train_idx][[col]], 
                   df.iloc[train_idx][target])
        encoded[test_idx] = encoder.transform(
            df.iloc[test_idx][[col]]
        ).values.flatten()
    
    return encoded
```

---

## 4. Calendar Feature Engineering

### Key Findings

**Calendar features are the single most important feature type for short time series.** They provide seasonal information at zero data cost.

**Evidence**:
- Microsoft AutoML: "Calendar features can help regression models learn seasonal patterns at several cadences"
- Kaggle competitions: "Information on promotions, holidays, and events proved highly useful"
- Weather and calendar data are "50% or more significant when compared with other exogenous variables" (research paper on energy forecasting)
- AutoML generates calendar features from datetime columns automatically

**Essential Calendar Features**:

| Feature Type | Examples | Implementation |
|---|---|---|
| Basic cyclical | dayofweek, month, dayofyear, quarter | `np.sin(2 * np.pi * df['dayofweek'] / 7)` |
| Holiday indicators | IsHoliday, DaysToHoliday, DaysAfterHoliday | Use `holidays` library |
| Event proximity | days_until_holiday, days_since_holiday | Custom calculation |
| Business calendar | is_weekend, is_month_start, is_month_end | pandas `.dt` accessor |
| Custom events | promotion_period, sale_event, product_launch | Domain-specific |

**Key Implementation Details**:

```python
def create_calendar_features(df, date_col):
    """Create comprehensive calendar features."""
    df = df.copy()
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['dayofmonth'] = df[date_col].dt.day
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = (df['dayofmonth'] <= 5).astype(int)
    df['is_month_end'] = (df['dayofmonth'] >= 25).astype(int)
    
    # Cyclical encoding - critical for continuity
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Proximity to holidays
    import holidays
    us_holidays = holidays.US(years=df[date_col].dt.year.unique())
    holiday_dates = pd.to_datetime(list(us_holidays.keys()))
    
    df['days_to_nearest_holiday'] = df[date_col].apply(
        lambda x: min((holiday_dates - x).days) if len(holiday_dates) > 0 else 0
    )
    
    return df
```

### Suitability for <100 Days: **EXCELLENT**

Calendar features require zero historical data and are universally applicable. They are the #1 priority for short series.

### Implementation Complexity: **Low**

---

## 5. Lag Feature Selection

### Key Findings

**The Problem**: With <100 days, every lag feature costs data rows (NaN at the beginning). Choosing the right lags is critical.

**Evidence from Research**:
- "PACF quantifies the correlation of a time series with its lags... The examination of the PACF plot can be used to guide the lag selection process" (Lag Selection for Univariate Time Series, arXiv 2024)
- Common heuristics tested in research: PACF-based, False Nearest Neighbors (FNN), Frequency-based, Horizon-based
- "For short time series, the maximum number of lags is limited by a quarter of the series"

**Lag Selection Methods**:

| Method | How It Works | Suitability for <100 Days |
|---|---|---|
| PACF | Select lags where PACF exceeds significance threshold | Good - data-driven |
| Domain knowledge | Weekly pattern -> lags 7, 14, 21 | Good - reliable with little data |
| Frequency-based | Set lags to seasonal period (7 for daily) | Good - simple heuristic |
| Cross-validation | Test different lag sets, select best | Limited - needs sufficient data |

**Recommended Lags for <100 Days**:

```python
# Conservative lag selection for short series
def select_lags_for_short_series(df, target_col, max_lag_ratio=0.1):
    """
    Select lags based on series length.
    For 90 days, max_lag = 9 (very conservative).
    """
    n = len(df)
    max_lag = max(1, int(n * max_lag_ratio))
    
    # Always include lag 1 (most recent past)
    essential_lags = [1]
    
    # Add weekly lags if data permits (for daily data with weekly seasonality)
    if max_lag >= 7:
        essential_lags.append(7)
    if max_lag >= 14:
        essential_lags.append(14)
    
    # Use PACF for additional lags if enough data
    if n > 30:
        from statsmodels.tsa.stattools import pacf
        pacf_vals = pacf(df[target_col].dropna(), nlags=max_lag, method='ols')
        # Add lags where |PACF| > 0.1 (relaxed threshold for short series)
        for i in range(2, len(pacf_vals)):
            if abs(pacf_vals[i]) > 0.1 and i not in essential_lags:
                essential_lags.append(i)
    
    return sorted(essential_lags)
```

**For 90 Days of Daily Data**:
- **Recommended lags**: [1, 7, 14] - short and weekly
- **Avoid**: Lags > 21 (consume too much data, unreliable)
- **Data cost**: Lag 14 loses 14 rows (15% of 90 days)

### Suitability for <100 Days: **Moderate** - use short lags only

### Implementation Complexity: **Low**

---

## 6. Target Encoding for Time Series Categoricals

### Key Findings

**The Problem**: Standard target encoding leaks target information if done globally. For time series, temporal awareness is critical.

**Safe Temporal Target Encoding**:
1. **Expanding window encoding**: Only use data up to current point to encode
2. **Time-based CV encoding**: Fit encoder on past folds only
3. **Smoothing**: Blend category mean with global mean to handle rare categories
4. **Add noise**: Add small random noise to prevent overfitting

**Evidence**:
- Kaggle 1st place solution (Insurance competition): "A common way to improve CV/LB score is to provide various encodings for categorical features... label encoding, target encoding mean, count encoding, TE median, TE min, TE max, TE nunique"
- Data leakage prevention: "Encoding categorical variables with target encoding computed on the full dataset" is a classic leakage trap

**Recommended Implementation**:

```python
class TemporalTargetEncoder:
    """
    Target encoder that respects temporal order.
    Uses expanding window to prevent leakage.
    """
    def __init__(self, smoothing=10, min_samples=5):
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.global_mean = None
        self.category_stats = {}  # Running stats per category
        
    def fit(self, df, cat_col, target_col, date_col):
        """Build expanding window statistics."""
        df = df.sort_values(date_col)
        self.global_mean = df[target_col].mean()
        
        # Compute running statistics per category
        for cat in df[cat_col].unique():
            mask = df[cat_col] == cat
            cat_data = df.loc[mask].sort_values(date_col)
            
            self.category_stats[cat] = {
                'cumulative_means': cat_data[target_col].expanding(
                    min_periods=self.min_samples
                ).mean().values,
                'cumulative_counts': cat_data[target_col].expanding(
                    min_periods=1
                ).count().values,
                'dates': cat_data[date_col].values
            }
    
    def transform(self, df, cat_col, date_col):
        """Get encoded value using only past data."""
        encoded = np.full(len(df), self.global_mean)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            cat = row[cat_col]
            date = row[date_col]
            
            if cat in self.category_stats:
                stats = self.category_stats[cat]
                # Find the latest entry before current date
                past_mask = stats['dates'] < date
                if past_mask.sum() >= self.min_samples:
                    mean = stats['cumulative_means'][past_mask][-1]
                    count = stats['cumulative_counts'][past_mask][-1]
                    # Shrinkage toward global mean
                    encoded[idx] = (mean * count + self.global_mean * self.smoothing) / (
                        count + self.smoothing
                    )
        
        return encoded
```

### Suitability for <100 Days: **HIGH** - especially with hierarchical smoothing

For short series, category-level statistics are noisy. Smooth heavily toward the parent-category mean.

### Implementation Complexity: **Medium** (requires temporal-aware implementation)

---

## 7. Automated Feature Extraction (TSFresh)

### Key Findings

**TSFresh** (Time Series Feature extraction based on scalable hypothesis tests) automatically extracts 794+ features from time series data.

**Feature Set Includes**:
- Basic statistics: mean, std, max, min, kurtosis, skewness
- Time-series specific: autocorrelation, trend, seasonality measures
- Complexity features: entropy, LLE, sample entropy
- Frequency domain: FFT coefficients, spectral features

**Key Papers**:
- Christ et al. (2018): TSFresh uses "hypothesis tests to mathematically control the percentage of irrelevant extracted features"
- Empirical evaluation (2021): TSFresh extracts 794 features, takes ~2.5s per 1000-sample series

**For Short Time Series (<100 days)**:

| Aspect | Assessment |
|---|---|
| Number of features | 794 features from 90 data points = severe overfitting risk |
| Built-in selection | Yes - statistical hypothesis testing filters irrelevant features |
| Recommendation | **Use with extreme caution** - limit feature subset |
| Best practice | Extract only from rolling windows, not full series |

**Safer Approach for Short Data**:
```python
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

# Use minimal feature set for short series
settings = MinimalFCParameters()  # Only ~20 features instead of 794

# Extract from rolling windows only
features = extract_features(
    df,
    column_id='series_id',
    column_sort='date',
    default_fc_parameters=settings,
    n_jobs=4
)
```

**Alternative: Catch22** - 22 canonical features selected from 4,791 candidates:
- "~1000x faster than full hctsa feature set"
- "Near linear scaling with time-series length"
- Better suited for short series than TSFresh's full feature set

### Suitability for <100 Days: **LOW to MODERATE**

- Full TSFresh: **Not recommended** - too many features, high overfitting risk
- Minimal settings or Catch22: **Usable with care**
- Best approach: Extract from rolling windows, not the full series

### Implementation Complexity: **Medium**

---

## 8. Feature Engineering for Sparse Time Series

### Key Findings

**Sparse/Intermittent Demand**: When the time series has many zero-demand periods mixed with occasional non-zero demands. Common in retail (slow-moving items, spare parts).

**Key Research Finding (2024)**:
"Statistically grounded feature engineering is the dominant driver of performance" for intermittent demand. "A single-stage LightGBM regressor augmented with SHOS features achieves performance comparable to, or better than, a more complex two-stage hurdle architecture."

**SHOS (Smoothed Hybrid Occurrence Size) Algorithm**:
- Computes smoothed demand occurrence probability
- Computes smoothed conditional demand size
- Adapts smoothing strength based on series-level intermittency

**Essential Features for Sparse Demand**:

| Feature | Description | Why It Helps |
|---|---|---|
| Demand occurrence flag | Binary: was demand > 0? | Separates timing from magnitude |
| Time since last demand | Days since last non-zero | Captures demand frequency |
| Rolling zero ratio | Fraction of zeros in window | Measures intermittency |
| Croston features | Smoothed inter-demand interval & size | Specifically designed for this |
| Days since last sale | Recency feature | Captures demand reactivation |

```python
def create_sparse_demand_features(df, target_col):
    """Features specifically for intermittent/sparse demand."""
    df = df.copy()
    
    # Demand occurrence
    df['demand_occurred'] = (df[target_col] > 0).astype(int)
    
    # Time since last demand (forward fill)
    df['days_since_demand'] = (
        df['demand_occurred'].replace(0, np.nan)
        .groupby((df['demand_occurred'] == 1).cumsum())
        .cumcount()
    )
    df['days_since_demand'] = df['days_since_demand'].ffill().fillna(999)
    
    # Rolling zero ratio
    for window in [7, 14, 30]:
        df[f'zero_ratio_{window}d'] = (
            (df[target_col] == 0).rolling(window, min_periods=1).mean()
        )
    
    # Count of non-zero periods
    for window in [7, 14, 30]:
        df[f'nonzero_count_{window}d'] = (
            df['demand_occurred'].rolling(window, min_periods=1).sum()
        )
    
    # Expanding cumulative features
    df['expanding_mean_positive'] = (
        df.loc[df[target_col] > 0, target_col]
        .expanding().mean()
        .reindex(df.index).ffill().fillna(0)
    )
    
    return df
```

### Suitability for <100 Days: **HIGH**

Sparse demand features (occurrence flags, recency) require very little data to compute and are extremely informative.

### Implementation Complexity: **Medium**

---

## 9. Rolling Window Best Practices

### Key Findings

**Critical Rule**: Always shift by 1 before rolling to prevent data leakage.

```python
# WRONG - includes current value in the window
df['roll_mean_7'] = df['sales'].rolling(7).mean()  # LEAKAGE!

# CORRECT - only uses past values
df['roll_mean_7'] = df['sales'].shift(1).rolling(7).mean()  # Safe
```

**Best Practices from Research**:

1. **Window sizes**: "Different characteristics need different window sizes. Short windows (3-5) and long windows (14-30)" (Analytics Vidhya)
2. **Multiple statistics**: mean, std, min, max, median
3. **Exponentially weighted**: EWM gives more weight to recent observations
4. **Seasonal rolling**: Same day of previous weeks (e.g., last 4 Mondays)

**For <100 Days**:

| Window Size | Suitable? | Notes |
|---|---|---|
| 3 days | Yes | Very recent trend |
| 7 days | Yes | Full week captures weekly pattern |
| 14 days | Yes with care | Needs 15+ days of history |
| 30 days | Marginal | Uses 1/3 of available data |
| >30 days | No | Insufficient data for reliable statistics |

**Key Rolling Features**:

```python
def create_rolling_features(df, target_col, max_window=14):
    """Create rolling features safe for short time series."""
    df = df.copy()
    shifted = df[target_col].shift(1)  # CRITICAL: prevent leakage
    
    windows = [3, 7]
    if max_window >= 14:
        windows.append(14)
    
    for w in windows:
        df[f'roll_mean_{w}'] = shifted.rolling(w, min_periods=1).mean()
        df[f'roll_std_{w}'] = shifted.rolling(w, min_periods=1).std()
        df[f'roll_min_{w}'] = shifted.rolling(w, min_periods=1).min()
        df[f'roll_max_{w}'] = shifted.rolling(w, min_periods=1).max()
        
        # EWM gives more weight to recent observations
        df[f'ewm_mean_{w}'] = shifted.ewm(span=w, min_periods=1).mean()
    
    # Trend features
    df['roll_mean_7_vs_roll_mean_3'] = (
        df['roll_mean_7'] - df['roll_mean_3']
    )
    
    # Expanding window (uses all available history)
    df['expanding_mean'] = shifted.expanding(min_periods=1).mean()
    df['expanding_std'] = shifted.expanding(min_periods=1).std()
    
    return df
```

**Seasonal Rolling (Same Day of Week)**:
```python
# Capture "last 4 Mondays" instead of "last 28 days"
df['seasonal_roll_mean'] = (
    df.groupby(df['date'].dt.dayofweek)[target_col]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)
```

### Suitability for <100 Days: **MODERATE** - use small windows only

### Implementation Complexity: **Low** (but easy to get wrong with leakage)

---

## 10. Entity Embeddings for Time Series

### Key Findings

**Entity embeddings** learn dense vector representations of categorical variables. Popularized in the Rossmann Kaggle competition (3rd place used categorical embeddings in neural networks).

**How They Work**:
- Each category (e.g., product ID) is mapped to a learnable vector
- Similar categories get similar vectors
- Dimensionality is much lower than one-hot encoding
- Can be trained end-to-end with the forecasting model

**Deep Learning Architectures Using Embeddings**:

| Architecture | Embedding Usage | Best For |
|---|---|---|
| DeepAR | Categorical covariates as embeddings | Probabilistic forecasting, many series |
| TFT | Variable selection with embeddings | Interpretable multi-horizon forecasting |
| N-BEATS/N-BEATSx | No embeddings (pure time series) | Univariate with many data points |
| TabTransformer | Attention over categorical embeddings | Tabular + time series hybrid |

**For Short Time Series**:
- Entity embeddings are **ideal** because they learn across all series
- A product with limited history still gets a meaningful embedding from shared patterns
- Embedding dimension: `min(50, (n_categories + 1) // 2)` - keep small for limited data
- Can pre-train embeddings on related tasks if available

**Hybrid Approach** (recommended):
1. Train embeddings with a neural network (DeepAR, TFT)
2. Extract trained embeddings
3. Use as features in gradient-boosted trees (XGBoost/LightGBM)

```python
# PyTorch Forecasting: Complete example with embeddings
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# Define dataset with categorical embeddings
dataset = TimeSeriesDataSet(
    df,
    time_idx='time_idx',
    target='sales',
    group_ids=['product_id', 'store_id'],
    static_categoricals=['product_category', 'store_type'],
    time_varying_known_categoricals=['month', 'dayofweek'],
    time_varying_known_reals=['time_idx'],
    time_varying_unknown_reals=['sales'],
    max_encoder_length=30,  # Lookback window
    max_prediction_length=7,  # Forecast horizon
    categorical_encoders={'product_id': NaNLabelEncoder().fit(df['product_id'])}
)

# The model learns embeddings for all categorical variables automatically
tft = TemporalFusionTransformer.from_dataset(
    dataset,
    hidden_size=16,  # Keep small for limited data
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    learning_rate=0.01
)
```

### Suitability for <100 Days: **HIGH** - especially for high-cardinality categoricals

### Implementation Complexity: **Medium-High** (requires deep learning framework)

---

## Summary: Feature Engineering Recommendations for ~90 Days of Data

### Priority Ranking (Highest to Lowest Impact)

| Priority | Technique | Impact | Complexity | Data Cost |
|---|---|---|---|---|
| 1 | Calendar features (cyclical encoding) | Very High | Low | None |
| 2 | Cross-series pooled statistics | Very High | Medium | None |
| 3 | Short lag features (1, 7, 14) | High | Low | 14 rows |
| 4 | Target encoding (temporal-aware) | High | Medium | None |
| 5 | Small rolling windows (3, 7) | High | Low | 7 rows |
| 6 | Entity embeddings (for categoricals) | High | High | None |
| 7 | Sparse demand features (if applicable) | High | Medium | None |
| 8 | Holiday proximity features | Medium | Low | None |
| 9 | Expanding window features | Medium | Low | None |
| 10 | Seasonal same-day-of-week features | Medium | Low | None |
| 11 | Catch22 (minimal subset) | Low-Medium | Medium | None |
| 12 | Decomposition features (STL) | Low-Medium | Medium | Needs >2 cycles |
| 13 | Full TSFresh | Not recommended | Medium | Overfitting risk |
| 14 | Long lags (>21) | Not recommended | Low | Too much data loss |

### Critical Implementation Rules

1. **Always shift(1) before rolling()** - prevent data leakage
2. **Use temporal-aware target encoding** - never encode using future data
3. **Smooth categorical encodings** heavily toward parent-group means
4. **Keep embedding dimensions small** - limited data can't support large embeddings
5. **Validate with time-series CV** - never random train/test split
6. **Monitor for overfitting** - with 90 days, less is often more

### Model-Specific Recommendations

| Model Type | Feature Strategy |
|---|---|
| LightGBM/XGBoost | Calendar + lags + rolling + target-encoded categoricals + cross-series stats |
| DeepAR | Minimal manual features; let model learn from raw series + categorical embeddings |
| TFT | Calendar + embeddings + variable selection handles feature importance |
| N-BEATS | Very limited; model learns basis functions from raw series |
| Linear models | Heavy calendar encoding + short lags + trend features |

---

## References

1. "Learnings from Kaggle's Forecasting Competitions" - arXiv:2009.07701
2. "Feature Engineering Automation for Time Series Analysis" - Lisbon Technical University
3. "Evaluation of ML and DL Models for Short Time Series" - TU Wien, 2025
4. "Primacy of Feature Engineering over Architectural Complexity for Intermittent Demand" - PMC 2024
5. "Lag Selection for Univariate Time Series using Deep Learning" - arXiv:2405.11237
6. "TSFresh: Time Series Feature Extraction" - Christ et al., Neurocomputing 2018
7. "Catch22: Canonical Time-series Characteristics" - Lubba et al., 2019
8. "Multi-dataset LSTM Analysis for Supply Chain Forecasting" - Springer 2026
9. "Temporal Fusion Transformer" - Lim et al., 2021
10. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" - Salinas et al., 2020
11. "Calendar Features for Time Series Forecasting in AutoML" - Microsoft, 2025
12. "N-BEATSx: Neural Basis Expansion with Exogenous Variables" - Olivares et al., 2023
