# Dimension 11: Automated Feature Engineering for Time Series

## Research Summary

This document provides a comprehensive analysis of automated feature engineering (AutoFE) tools for demand forecasting with limited historical data (~2 months), multi-table structure (order header + order line + operation log), and the critical need to avoid overfitting. We examine TSFresh, Featuretools, AutoFeat, BorutaPy, recursive feature elimination, and genetic programming approaches, with evidence-based recommendations.

---

## 1. Key Findings

### Finding 1: TSFresh is the most mature AutoFE tool for time series but requires careful configuration with limited data

**Claim:** TSFresh extracts ~794 features automatically and includes the FRESH algorithm for statistical feature selection using the Benjamini-Yekutieli procedure, but with only 2 months of data, the full feature set will overfit without aggressive filtering [^634^] [^749^].
**Source:** TSFresh Documentation; Bake Off Redux Paper (arXiv)
**URL:** https://tsfresh.readthedocs.io/en/latest/text/forecasting.html; https://arxiv.org/pdf/2304.13029v2
**Date:** 2024; 2023
**Excerpt:** "TSFresh is a collection of just under 800 features extracted from time series. While the features can be used on their own, a feature selection method called FRESH is provided to remove irrelevant features. FRESH considered each feature using multiple hypotheses tests... The Benjamini-Yekutieli procedure is then used to control the false discovery rate."
**Confidence:** High

### Finding 2: Featuretools excels at multi-table relational feature engineering with proper temporal safeguards

**Claim:** Featuretools' Deep Feature Synthesis (DFS) can generate cross-table aggregation features (e.g., avg processing time per warehouse, order frequency per customer) using aggregation and transform primitives, with cutoff_time parameter preventing data leakage [^643^] [^765^].
**Source:** Featuretools Documentation; Reintech Blog
**URL:** https://featuretools.alteryx.com/en/stable/guides/time_series.html; https://reintech.io/blog/feature-engineering-automation-featuretools-feature-store
**Date:** 2024; 2026
**Excerpt:** "Featuretools supports this through cutoff times... This approach ensures your training features match what will be available during inference, preventing the common pitfall of leaking future information into your model."
**Confidence:** High

### Finding 3: AutoFeat discovers mathematical combinations but is unsuitable for small time series datasets

**Claim:** AutoFeat generates non-linear feature combinations (products, ratios, dimensionless quantities via Buckingham Pi theorem) but requires data without NaNs and internally uses LassoLarsCV which can overfit with limited samples [^668^] [^681^].
**Source:** AutoFeat Paper (arXiv); PyPI Documentation
**URL:** https://ar5iv.labs.arxiv.org/html/1901.07329; https://pypi.org/project/autofeat/2.1.2/
**Date:** 2019; 2023
**Excerpt:** "The AutoFeatRegressor, AutoFeatClassifier, and FeatureSelector models need to be fit on data without NaNs, as they internally call the sklearn LassoLarsCV model."
**Confidence:** High

### Finding 4: BorutaPy outperforms RFE in comprehensive feature selection benchmarks

**Claim:** Boruta consistently achieved the best balance between feature reduction and model performance across multiple comparative studies, with RFE ranking second [^734^] [^740^].
**Source:** MDPI - Diabetes Prediction; MDPI - Soil Organic Carbon Mapping
**URL:** https://www.mdpi.com/2075-4418/15/20/2622; https://www.mdpi.com/2073-445X/14/4/677
**Date:** 2025; 2025
**Excerpt:** "Among the feature selection methods evaluated, Boruta consistently achieved the best balance between reducing the number of features and maintaining or improving model performance."
**Confidence:** High

### Finding 5: With only 2 months of data, aggressive feature selection is mandatory

**Claim:** The curse of dimensionality makes automated feature engineering dangerous with limited data. Research shows that wrapper methods (RFE, Boruta) can overfit small datasets, and a Boruta-then-RFE two-stage approach may be optimal [^644^] [^782^].
**Source:** Feature Engineering Automation Thesis; PMC - Consensus Features Nested CV
**URL:** https://fenix.tecnico.ulisboa.pt/downloadFile/1689244997263130/83473-Helio-Domingos_dissertacao.pdf; https://pmc.ncbi.nlm.nih.gov/articles/PMC8801637/
**Date:** 2024; 2021
**Excerpt:** "Composition features have more features than the other sets of analysis. This could lead to overfit and increase the complexity of the model... wrapper methods can be subject to overfitting as model complexity increases."
**Confidence:** High

### Finding 6: Kaggle forecasting competitions confirm rolling statistics + groupby aggregations as the winning pattern

**Claim:** Winners of Rossmann, Corporacion Favorita, and M5 competitions all relied on rolling statistics (moving averages, medians) grouped by hierarchical factors (store, item, class) combined with event counters - features that Featuretools can automate [^4^] [^7^].
**Source:** Learnings from Kaggle's Forecasting Competitions (AAU); NVIDIA Grandmaster Pro Tip
**URL:** https://vbn.aau.dk/ws/files/483966133/Kaggle_3_.pdf; https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/
**Date:** 2024; 2025
**Excerpt:** "The most powerful feature engineering technique is groupby aggregations. Namely, we execute the code `groupby(COL1)[COL2].agg(STAT)`... The best 500 discovered features significantly boosted the accuracy of my XGBoost model."
**Confidence:** High

---

## 2. Major Techniques

### 2.1 TSFresh: Time Series Feature Extraction Based on Scalable Hypothesis Tests

TSFresh is the most comprehensive open-source time series feature extraction library, developed by Blue Yonder. It extracts ~794 features across multiple categories:

**Feature Categories:**
- **Descriptive statistics:** mean, std, variance, min, max, median, quantiles
- **Absolute energy:** sum of squares of values
- **Spectral features:** FFT coefficients, spectral centroid, spectral entropy
- **Autocorrelation:** at multiple lags
- **Trend features:** linear trend, augmented Dickey-Fuller test
- **Complexity features:** Lempel-Ziv complexity, cid_ce (complexity invariant distance)
- **Shape features:** number of peaks, number of crossings, longest strike
- **Entropy features:** sample entropy, permutation entropy
- **Distribution features:** skewness, kurtosis

**The FRESH Algorithm (3 phases) [^703^] [^761^]:**
1. **Feature Extraction:** Calculate all ~794 features for each time series
2. **Feature Significance Testing:** Each feature is individually tested for significance using:
   - Fisher's exact test (binary target, binary feature)
   - Mann-Whitney U test (binary target, real feature)
   - Kendall's tau rank test (real target, real feature)
   - Kolmogorov-Smirnov test (distribution comparison)
3. **Multiple Test Procedure:** Benjamini-Yekutieli procedure controls false discovery rate

**For Forecasting [^634^]:**
TSFresh uses a rolling mechanism where features are extracted on sub-time series up to time t to predict t+1:
```python
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame

# Roll the time series to create sub-series for each time point
df_rolled = roll_time_series(df, column_id="id", column_sort="time",
                              max_timeshift=10,  # window size
                              rolling_direction=1)

# Extract features on rolled data
df_features = extract_features(df_rolled, column_id="id", column_sort="time")

# Filter relevant features
from tsfresh import select_features
X_selected = select_features(df_features, y, fdr_level=0.05)
```

**Efficiency Presets [^764^] [^783^]:**
```python
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

# For limited data, use minimal features (fastest, least risk of overfitting)
# EfficientFCParameters = subset proven useful in benchmarks
# ComprehensiveFCParameters = all 794 features (overfit risk)
```

### 2.2 Featuretools: Deep Feature Synthesis for Multi-Table Data

Featuretools automates feature engineering across relational tables using **Deep Feature Synthesis (DFS)**. It is particularly suited for our order header + order line + operation log structure.

**Core Concepts [^680^] [^758^]:**
- **EntitySet:** Collection of related data tables
- **Feature Primitives:** Building blocks for feature generation
  - *Aggregation Primitives:* sum, mean, std, count, max, min, nunique, mode, median
  - *Transform Primitives:* absolute, square root, time_since_previous, hour, day, month
- **DFS:** Recursively stacks primitives to create complex features up to `max_depth`

**For Our Multi-Table Structure:**
```python
import featuretools as ft

# Create EntitySet
es = ft.EntitySet(id="orders")

# Add order header table
es = es.add_dataframe(dataframe_name="orders",
                      dataframe=df_orders,
                      index="order_id",
                      time_index="order_date")

# Add order line table
es = es.add_dataframe(dataframe_name="order_lines",
                      dataframe=df_lines,
                      index="line_id",
                      time_index="created_at")

# Add operation log table
es = es.add_dataframe(dataframe_name="operations",
                      dataframe=df_ops,
                      index="op_id",
                      time_index="operation_date")

# Define relationships
es = es.add_relationship("orders", "order_id", "order_lines", "order_id")
es = es.add_relationship("orders", "order_id", "operations", "order_id")

# Generate features with temporal safety
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="orders",
    cutoff_time=cutoff_times,  # CRITICAL: prevents data leakage
    agg_primitives=["mean", "std", "count", "max", "min", "sum",
                    "avg_time_between", "trend"],
    trans_primitives=["time_since_previous", "day", "month", "weekday"],
    max_depth=2  # Don't go too deep with limited data
)
```

**Critical for Forecasting: Cutoff Times [^770^] [^785^]:**
```python
# Cutoff times prevent using future data
import pandas as pd
cutoff_times = pd.DataFrame({
    'order_id': order_ids,
    'time': forecast_dates,  # Only use data <= this timestamp
    'target': target_values   # Optional: pass target through
})

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="orders",
    cutoff_time=cutoff_times,
    cutoff_time_in_index=True  # Ensures temporal safety
)
```

**Cross-Table Feature Examples for Our Problem:**
- `MEAN(order_lines.quantity)` - average quantity per order
- `AVG_TIME_BETWEEN(operations.operation_date)` - avg time between operations
- `TREND(operations.processing_time, operation_date)` - trend in processing time
- `COUNT(order_lines WHERE status = 'delayed')` - count of delayed lines
- `STD(order_lines.quantity)` - variability in order quantities

### 2.3 AutoFeat: Symbolic Feature Engineering

AutoFeat generates non-linear features through symbolic mathematics [^668^] [^679^].

**How it works:**
1. Creates polynomial, logarithmic, exponential, and trigonometric combinations
2. Uses Buckingham Pi theorem for dimensionless quantities
3. Selects features via LassoLarsCV regularized regression
4. Trains linear model on selected features

**Limitations for Our Problem:**
- Requires complete data (no NaNs)
- LassoLarsCV overfits easily with small sample sizes
- Designed for tabular, not time series data
- Cannot handle multi-table relational structures directly
- Feature explosion: d original features -> O(d^2) combinations

**Usage Pattern:**
```python
from autofeat import AutoFeatRegressor

model = AutoFeatRegressor(
    feat_sel_steps=1,  # Reduce for small data
    max_gb=1,          # Limit generated features
    units={}           # Optional: physical units
)
X_features = model.fit_transform(X, y)
```

### 2.4 BorutaPy: All-Relevant Feature Selection

Boruta is a wrapper method that identifies ALL features relevant to the target, unlike other methods that find a minimal subset [^695^] [^710^].

**Algorithm:**
1. Creates "shadow features" by shuffling each real feature
2. Trains a tree-based model on real + shadow features
3. Compares real feature importance to shadow feature importance maximum
4. Features significantly more important than shadows are "confirmed"
5. Features significantly less important are "rejected"
6. Undecided features may require more iterations

**Implementation [^695^]:**
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

# BorutaPy requires numpy arrays
X_np = X.values
y_np = y.values.ravel()

# Use RandomForest as base estimator
rf = RandomForestRegressor(
    n_estimators='auto',  # Auto-determines number of trees
    max_depth=5,          # Limit depth to prevent overfitting
    random_state=42
)

feat_selector = BorutaPy(
    rf,
    n_estimators='auto',
    max_iter=100,
    random_state=42,
    verbose=2
)

feat_selector.fit(X_np, y_np)

# Results
confirmed_features = X.columns[feat_selector.support_].tolist()
tentative_features = X.columns[feat_selector.support_weak_].tolist()
ranking = feat_selector.ranking_

# Transform data
X_selected = feat_selector.transform(X_np)
```

**With LightGBM (faster for large feature sets) [^626^]:**
```python
import lightgbm as lgb
from boruta import BorutaPy

# LightGBM as base model (faster for many features)
gbm = lgb.LGBMRegressor(n_estimators=100, max_depth=5)
feat_selector = BorutaPy(gbm, n_estimators='auto')
```

### 2.5 Recursive Feature Elimination (RFE)

RFE systematically removes the weakest features according to model-derived feature importances [^648^] [^702^].

**Standard RFE:**
```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# Basic RFE with fixed number of features
rfe = RFE(estimator=RandomForestRegressor(max_depth=5),
          n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# RFE with Cross-Validation (recommended)
tscv = TimeSeriesSplit(n_splits=3)
rfecv = RFECV(
    estimator=RandomForestRegressor(max_depth=5),
    step=0.1,           # Remove 10% at each iteration
    cv=tscv,            # Time series cross-validation
    scoring='neg_mean_absolute_error',
    min_features_to_select=5
)
rfecv.fit(X, y)
optimal_n = rfecv.n_features_
```

**Two-Stage Boruta + RFE Approach [^747^]:**
```python
# Stage 1: Boruta for initial screening
boruta = BorutaPy(RandomForestRegressor(), n_estimators='auto')
boruta.fit(X.values, y.values)
X_boruta = X.loc[:, boruta.support_]

# Stage 2: RFE for further refinement
rfecv = RFECV(estimator=RandomForestRegressor(), cv=tscv)
rfecv.fit(X_boruta, y)
X_final = rfecv.transform(X_boruta)
```

### 2.6 Genetic Programming: gplearn SymbolicTransformer

gplearn uses genetic programming to evolve mathematical expressions that create new features [^692^] [^698^].

**SymbolicTransformer for Feature Engineering:**
```python
from gplearn.genetic import SymbolicTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Standardize first (important for genetic programming)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create symbolic features
gp = SymbolicTransformer(
    generations=20,              # Fewer generations for small data
    population_size=1000,        # Population of candidate expressions
    hall_of_fame=50,             # Best expressions to keep
    n_components=10,             # Number of new features to generate
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                   'abs', 'neg', 'inv'],  # Safe functions
    parsimony_coefficient=0.001, # Penalize complex expressions
    max_samples=0.9,             # Use 90% of data per generation
    random_state=42,
    n_jobs=-1
)

# Fit and transform
gp.fit(X_scaled, y)
X_gp = gp.transform(X_scaled)

# Combine original + genetic features
X_enhanced = np.hstack([X_scaled, X_gp])

# Train model on enhanced features
model = Ridge()
model.fit(X_enhanced, y)
```

**Performance Example from Documentation [^698^]:**
| Model | Features | R^2 Score |
|-------|----------|-----------|
| Ridge | Original only | 0.434 |
| Ridge | Original + SymbolicTransformer | 0.534 |

**Warning for Small Datasets:** With only 2 months of data, genetic programming is high-risk due to:
- Large search space relative to sample size
- Overfitting to training data patterns
- Long computation time relative to benefit

### 2.7 TPOT: Genetic Programming for Full Pipeline Optimization

TPOT automates both feature engineering AND model selection using genetic programming [^651^] [^653^].

```python
from tpot import TPOTRegressor

tpot = TPOTRegressor(
    generations=10,           # Reduce for small data
    population_size=50,       # Smaller population
    offspring_size=50,
    verbosity=2,
    max_time_mins=30,         # Limit search time
    scoring='neg_mean_absolute_error',
    cv=TimeSeriesSplit(n_splits=3),
    random_state=42
)
tpot.fit(X, y)

# Export best pipeline
print(tpot.fitted_pipeline_)
```

**Key Limitation [^653^]:** TPOT is not time-series aware and can easily leak future information. The TimeSeriesSplit helps but doesn't fully address the issue.

---

## 3. Implementation Details

### 3.1 Recommended Pipeline Architecture

```python
import pandas as pd
import numpy as np
import featuretools as ft
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# ============================================================
# PHASE 1: Multi-Table Feature Engineering (Featuretools)
# ============================================================
def generate_relational_features(df_orders, df_lines, df_ops, cutoff_times):
    """Generate cross-table features with temporal safety."""
    es = ft.EntitySet(id="orders")

    es = es.add_dataframe(dataframe_name="orders",
                          dataframe=df_orders,
                          index="order_id",
                          time_index="order_date")
    es = es.add_dataframe(dataframe_name="order_lines",
                          dataframe=df_lines,
                          index="line_id",
                          time_index="created_at")
    es = es.add_dataframe(dataframe_name="operations",
                          dataframe=df_ops,
                          index="op_id",
                          time_index="operation_date")

    es = es.add_relationship("orders", "order_id", "order_lines", "order_id")
    es = es.add_relationship("orders", "order_id", "operations", "order_id")

    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="orders",
        cutoff_time=cutoff_times,
        agg_primitives=["mean", "std", "count", "max", "min", "sum",
                        "avg_time_between", "trend", "num_unique"],
        trans_primitives=["time_since_previous", "day", "month",
                           "weekday", "is_weekend"],
        max_depth=2
    )
    return feature_matrix

# ============================================================
# PHASE 2: Time Series Feature Extraction (TSFresh - Limited)
# ============================================================
def generate_ts_features(df_rolled, target_col='quantity'):
    """Extract time series features with efficient preset."""
    # Use EfficientFCParameters instead of Comprehensive to limit overfitting
    settings = EfficientFCParameters()

    features = extract_features(
        df_rolled,
        column_id="id",
        column_sort="time",
        column_value=target_col,
        default_fc_parameters=settings,
        n_jobs=4,
        impute_function=impute  # Handle NaNs
    )
    return features

# ============================================================
# PHASE 3: Feature Selection (BorutaPy + RFE)
# ============================================================
def select_features_boruta_rfe(X, y, max_features=20):
    """Two-stage feature selection optimized for small datasets."""

    # Stage 1: Boruta for initial screening
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    boruta = BorutaPy(rf, n_estimators='auto', max_iter=50, random_state=42)
    boruta.fit(X.values, y.values)

    # Keep confirmed + tentative features
    mask_boruta = boruta.support_ | boruta.support_weak_
    X_boruta = X.loc[:, mask_boruta]

    # Stage 2: RFE for final selection (TimeSeries CV)
    if X_boruta.shape[1] > max_features:
        tscv = TimeSeriesSplit(n_splits=3)
        rfecv = RFECV(
            estimator=lgb.LGBMRegressor(n_estimators=100, max_depth=4),
            step=1,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            min_features_to_select=min(max_features, X_boruta.shape[1] // 2),
            n_jobs=-1
        )
        rfecv.fit(X_boruta, y)
        selected_cols = X_boruta.columns[rfecv.support_]
    else:
        selected_cols = X_boruta.columns

    return selected_cols
```

### 3.2 Feature Engineering for 2-Month Data: Conservative Settings

With only ~60 days of data, these are the critical parameter choices:

| Tool | Parameter | Recommended Value | Rationale |
|------|-----------|-------------------|-----------|
| TSFresh | `default_fc_parameters` | `EfficientFCParameters` | Subset proven useful; avoids ~700 irrelevant features |
| TSFresh | `fdr_level` | 0.01 (vs default 0.05) | More aggressive filtering |
| Featuretools | `max_depth` | 2 (not 3+) | Limits feature explosion |
| Featuretools | `agg_primitives` | Mean, std, count, trend | Core statistics only |
| BorutaPy | `max_depth` (base estimator) | 3-5 | Prevents base model overfitting |
| BorutaPy | `n_estimators` | 100-200 | Balance stability vs speed |
| RFECV | `min_features_to_select` | 5-10 | Hard floor to prevent underfitting |
| RFECV | `cv` | `TimeSeriesSplit(n_splits=3)` | Respects temporal ordering |

---

## 4. What Works

### 4.1 Groupby Aggregations: The Single Most Powerful Technique

**Evidence:** Every major Kaggle forecasting competition winner used `groupby(COL1)[COL2].agg(STAT)` patterns [^7^] [^4^].

**Winning Feature Patterns:**
- Rossmann winner: "statistics and their rolling versions calculated at different levels of the hierarchy, for different days of the week and promotion periods"
- Corporacion Favorita winner: "rolling statistics grouped by various factors such as store, item, class, and their combinations. The statistics used included measures of centrality and spread, as well as an exponential moving average"
- NVIDIA Grandmaster: "We use the speed of NVIDIA cuDF-Pandas to explore thousands of COL1, COL2, STAT combinations"

**For Our Problem:**
```python
# These groupby aggregations are high-value
features = {
    'avg_qty_per_warehouse': df.groupby('warehouse')['quantity'].mean(),
    'avg_qty_per_customer': df.groupby('customer')['quantity'].mean(),
    'avg_processing_time': df.groupby('warehouse')['processing_time'].mean(),
    'order_freq_per_customer_store': df.groupby(['customer', 'store'])['order_id'].count(),
    'std_qty_per_product': df.groupby('product')['quantity'].std(),
    'days_since_last_order': df.groupby('customer')['order_date'].diff().dt.days
}
```

### 4.2 Two-Stage Feature Selection (Boruta + RFE)

**Evidence:** Multiple studies show Boruta + RFE combination outperforms either alone [^729^] [^740^].

| Study | Best Method | Performance Gain |
|-------|-------------|-----------------|
| Soil Salinity (MDPI 2026) | RFE-RF | R^2 +0.28, RMSE -2.12 |
| Potato Nitrogen (MDPI 2025) | Boruta + SVM | R^2 0.546 vs 0.512 baseline |
| SOC Mapping (MDPI 2025) | Boruta + CNN-LSTM | R^2 = 0.80 (best overall) |

### 4.3 Rolling Statistics with Short Windows

**Evidence:** With 2 months of data, short windows work best [^644^].

From the Feature Engineering Automation thesis:
- **Smooth features (exponential moving averages)** performed best (MAE ~2.84)
- **Lag features** performed poorly with small dataset (MAE ~9.43)
- **Aggregation features** by week were better than by month
- **Composition features** overfit worst of all (MAE ~10.32, "could lead to overfit")

### 4.4 TSFresh with Efficient Settings

**Evidence:** Benchmarks show EfficientFCParameters achieves near-comprehensive performance [^666^] [^684^].

| Feature Set | Features | Time (1000-sample) | Accuracy |
|-------------|----------|-------------------|----------|
| catch22 | 22 | <10ms | Good |
| TSFEL | ~285 | 0.03s | Good |
| **TSFresh Efficient** | **~200** | **~0.5s** | **Best** |
| TSFresh Comprehensive | ~794 | 2.53s | Best (marginal gain) |

### 4.5 Cutoff Times for Temporal Safety

**Evidence:** Featuretools' cutoff_time is the only automated approach that guarantees no data leakage in multi-table forecasting [^770^] [^785^].

> "If your data has timestamps, the best way to prevent label leakage is to use a list of cutoff times, which specify the last point in time data is allowed to be used for each row in the resulting feature matrix." -- Featuretools FAQ

---

## 5. What Doesn't Work

### 5.1 AutoFeat on Small Time Series Datasets

**Claim:** AutoFeat generates too many features (O(d^2) combinations) and uses LassoLarsCV which overfits with limited samples [^668^] [^679^].
**Source:** AutoFeat Paper; Analytics India Magazine
**Confidence:** High

**Reasons to avoid:**
- Feature explosion: 20 original features -> 400+ combinations
- LassoLarsCV requires more samples than features for stability
- No built-in time series handling
- No multi-table support
- Cannot handle missing values (common in our data)

### 5.2 TPOT for Time Series Forecasting

**Claim:** TPOT is not time-series aware and requires extensive adaptation for forecasting [^653^].
**Source:** TPOT GitHub Issue
**URL:** https://github.com/EpistasisLab/tpot/issues/345
**Confidence:** High

**Reasons to avoid:**
- No native time series handling (no lag features, no rolling windows)
- Cross-validation is random, not temporal
- Computationally expensive for marginal benefit
- Pipeline evolution can leak future information

### 5.3 gplearn SymbolicTransformer with Very Limited Data

**Claim:** Genetic programming requires large populations and many generations to find meaningful expressions [^698^].
**Source:** gplearn Documentation
**Confidence:** High

**Risk factors with 2 months of data:**
- Population size 1000+ needed for diversity -> overfitting
- Evolved expressions memorize training noise
- Computation time better spent on domain-specific features

### 5.4 TSFresh Comprehensive Features Without Selection

**Claim:** Using all 794 features without filtering guarantees overfitting with n_samples < n_features [^644^].
**Source:** Feature Engineering Automation Thesis
**Confidence:** High

> "Composition features have more features than the other sets of analysis. This could lead to overfit and increase the complexity of the model." -- Thesis on NIO stock prediction

### 5.5 Wrapper Methods Without Proper Cross-Validation

**Claim:** RFE and Boruta can overfit when cross-validation doesn't respect temporal ordering [^766^] [^782^].
**Source:** GeeksforGeeks; PMC Consensus Features
**Confidence:** High

> "Wrapper methods can be subject to overfitting as model complexity increases." -- PMC Paper
> "Risk of overfitting: Fine-tuning features to a specific model can lead to an overfitted model." -- GeeksforGeeks

### 5.6 Deep Feature Synthesis Without max_depth Limit

**Claim:** DFS with max_depth > 2 generates exponentially more features, most of which are noise [^768^].
**Source:** Featuretools Optimization Guide
**Confidence:** High

> "Reducing max_depth: Overly deep features don't always provide better results and can dramatically increase computation. Typically, max_depth=2 or 3 captures the majority of useful information."

---

## 6. Competition Applications

### 6.1 M5 Forecasting Competition (2021)

**Claim:** Microsoft's AutoML achieved 99th percentile on M5 using automated feature engineering combined with Many Models approach [^687^] [^688^].
**Source:** Microsoft Tech Community Blog
**URL:** https://techcommunity.microsoft.com/blog/machinelearningblog/automated-machine-learning-on-the-m5-forecasting-competition/2933391
**Date:** 2021-11-17
**Excerpt:** "Engineered features associated with the calendar and a seasonal decomposition make the most impact on predictions. The seasonal decomposition is derived from weekly sales patterns detected by AutoML. Price is the most important of the original features."
**Key Features:**
- Calendar features (auto-engineered)
- Seasonal decomposition
- Price features
- 30,490 individual time-series models

### 6.2 Rossmann Store Sales (2015)

**Claim:** Winner used extensive feature engineering with groupby statistics at multiple hierarchy levels plus trend adjustment via ridge regression [^4^].
**Source:** Learnings from Kaggle's Forecasting Competitions (AAU)
**Excerpt:** "The main innovations on the feature front consisted of statistics and their rolling versions calculated at different levels of the hierarchy, for different days of the week and promotion periods."
**Key Features:**
- Average sales by product
- Moving averages of sales by product
- Average sales by product and promotion status
- Event counters (days until/into/after events)
- Weather information

### 6.3 Corporacion Favorita Grocery Sales (2018)

**Claim:** Winner used rolling statistics grouped by store, item, class, and combinations, with only recent data (1-5 months) despite 55 months available [^4^].
**Source:** Learnings from Kaggle's Forecasting Competitions
**Excerpt:** "The winner only used very recent data in the models, electing to drop older observations based on validation dataset performance. Thus, the final models used less than a full season of data for model fitting."
**Key Features:**
- Rolling statistics (mean, std, EMA)
- Grouped by store, item, class, combinations
- One model per forecast horizon (16 models total)
- Very recent data only (1-5 months)

### 6.4 Key Insight for Our Problem

The consistent pattern across all winning solutions:
1. **Rolling/groupby statistics** at multiple hierarchy levels
2. **Event counters** (time since/until events)
3. **Recent data focus** (even when more data is available)
4. **Validation-driven feature selection** (hold-out of forecast horizon length)

---

## 7. Recommended Approach

### 7.1 Tiered Recommendation Based on Data Size

| Tool | Recommendation | Priority | Risk Level |
|------|---------------|----------|------------|
| Featuretools | **HIGHLY RECOMMENDED** for cross-table features | P0 | Low (with cutoff times) |
| TSFresh Efficient | **RECOMMENDED** with strict FDR filtering | P1 | Medium |
| BorutaPy + RFE | **RECOMMENDED** two-stage selection | P1 | Low |
| Manual groupby | **REQUIRED** as baseline | P0 | Low |
| gplearn | **NOT RECOMMENDED** with < 60 days | P3 | High |
| AutoFeat | **NOT RECOMMENDED** for time series | P3 | High |
| TPOT | **NOT RECOMMENDED** without TS awareness | P3 | High |

### 7.2 Recommended Implementation Order

**Step 1: Manual groupby aggregations (Day 1)**
```python
# These require no autoFE tools and provide immediate value
groupby_features = [
    ('warehouse', 'quantity', ['mean', 'std', 'count']),
    ('customer_id', 'quantity', ['mean', 'std', 'count']),
    ('customer_id', 'days_to_deliver', ['mean', 'max']),
    ('product_id', 'quantity', ['mean', 'std']),
    ('store_id', 'quantity', ['mean', 'count']),
]
```

**Step 2: Featuretools relational features (Day 1-2)**
- Use EntitySet with order_lines and operations as child tables
- cutoff_time for temporal safety
- max_depth=2
- Core primitives: mean, std, count, trend, avg_time_between

**Step 3: TSFresh with Efficient preset (Day 2)**
- Apply to rolled time series of target variable
- fdr_level=0.01 (aggressive)
- Only on target + key numeric columns

**Step 4: BorutaPy screening (Day 3)**
- Combine Step 1+2+3 features
- RandomForest with max_depth=5
- Keep confirmed + tentative features

**Step 5: RFE refinement (Day 3)**
- Apply to Boruta-filtered features
- TimeSeriesSplit(n_splits=3)
- Target 10-20 final features

### 7.3 Overfitting Prevention Checklist

With 2 months of data, follow these rules:

1. **Feature count ceiling:** max(20, n_samples / 10) features
2. **Minimum window size:** At least 7 days for any rolling statistic
3. **Maximum TSFresh features:** Use EfficientFCParameters only
4. **Featuretools depth:** max_depth <= 2
5. **Boruta base model:** max_depth <= 5 for RandomForest
6. **Validation:** TimeSeriesSplit(n_splits=3) minimum
7. **Hold-out:** Keep final 7-14 days as true hold-out
8. **Regularization:** All final models should use L2 regularization

### 7.4 Expected Feature Categories for Our Problem

Based on Kaggle competition patterns and our multi-table structure:

**From Featuretools (cross-table):**
- `MEAN(order_lines.quantity)` - avg quantity per order
- `STD(order_lines.quantity)` - quantity variability
- `COUNT(order_lines)` - number of lines per order
- `AVG_TIME_BETWEEN(operations.op_date)` - operation frequency
- `TREND(operations.processing_time)` - processing time trend

**From groupby aggregations:**
- `warehouse_avg_qty_7d` - 7-day avg qty by warehouse
- `customer_avg_qty_14d` - 14-day avg qty by customer
- `customer_order_count_30d` - monthly order frequency
- `product_demand_trend` - linear trend in product demand

**From TSFresh (time series patterns):**
- `quantity__abs_energy` - sum of squares
- `quantity__c3__lag_1` - non-linearity metric
- `quantity__autocorrelation__lag_1` - 1-day autocorrelation
- `quantity__mean_second_derivative_central` - trend acceleration

---

## 8. Sources

| # | Source | URL | Date |
|---|--------|-----|------|
| 1 | TSFresh Documentation - Forecasting | https://tsfresh.readthedocs.io/en/latest/text/forecasting.html | 2024 |
| 2 | TSFresh Documentation - Large Data | https://tsfresh.readthedocs.io/en/latest/text/large_data.html | 2024 |
| 3 | TSFresh - Feature Filtering | https://tsfresh.readthedocs.io/en/v0.21.0/text/feature_filtering.html | 2024 |
| 4 | Featuretools - Time Series Guide | https://featuretools.alteryx.com/en/stable/guides/time_series.html | 2024 |
| 5 | Featuretools - API Reference | https://featuretools.alteryx.com/en/latest/api_reference.html | 2024 |
| 6 | Featuretools - FAQ (Data Leakage) | https://featuretools.alteryx.com/en/v1.0.0/resources/frequently_asked_questions.html | 2024 |
| 7 | AutoFeat Paper (arXiv) | https://ar5iv.labs.arxiv.org/html/1901.07329 | 2019 |
| 8 | AutoFeat PyPI | https://pypi.org/project/autofeat/2.1.2/ | 2023 |
| 9 | BorutaPy Documentation | https://danielhomola.com/boruta_py | 2026 |
| 10 | gplearn Documentation | https://gplearn.readthedocs.io/en/stable/ | 2026 |
| 11 | gplearn SymbolicTransformer Guide | https://deepwiki.com/trevorstephens/gplearn/4.3-feature-engineering-with-symbolictransformer | 2025 |
| 12 | RFE for Feature Selection (MLMastery) | https://www.machinelearningmastery.com/rfe-feature-selection-in-python/ | 2020 |
| 13 | Feature Engineering Automation Thesis | https://fenix.tecnico.ulisboa.pt/downloadFile/1689244997263130/83473-Helio-Domingos_dissertacao.pdf | 2024 |
| 14 | TSFresh Paper - FRESH Algorithm | https://ar5iv.labs.arxiv.org/html/1610.07717 | 2016 |
| 15 | Bake Off Redux - TSFresh Review | https://arxiv.org/pdf/2304.13029v2 | 2023 |
| 16 | Learnings from Kaggle Forecasting Competitions | https://vbn.aau.dk/ws/files/483966133/Kaggle_3_.pdf | 2024 |
| 17 | NVIDIA Grandmaster Pro Tip | https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/ | 2025 |
| 18 | M5 Competition - Microsoft AutoML | https://techcommunity.microsoft.com/blog/machinelearningblog/automated-machine-learning-on-the-m5-forecasting-competition/2933391 | 2021 |
| 19 | Featuretools with Cutoff Times | https://reintech.io/blog/feature-engineering-automation-featuretools-feature-store | 2026 |
| 20 | Boruta vs RFE Comparison (MDPI) | https://www.mdpi.com/2072-4292/18/6/955 | 2026 |
| 21 | Boruta Feature Selection (MDPI Diabetes) | https://www.mdpi.com/2075-4418/15/20/2622 | 2025 |
| 22 | Recursive Feature Elimination with CV (sklearn) | https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html | 2024 |
| 23 | Feature Selection with Annealing vs Boruta | https://link.springer.com/article/10.1186/s40854-024-00617-3 | 2024 |
| 24 | TSFresh Empirical Evaluation | https://arxiv.org/pdf/2110.10914 | 2021 |
| 25 | Featuretools DFS Guide | https://www.statology.org/how-to-automatically-generate-features-with-featuretools-using-dfs/ | 2025 |
| 26 | Featuretools PDF Documentation | https://featuretools.alteryx.com/_/downloads/en/v1.9.2/pdf/ | 2024 |
| 27 | TSFEL vs TSFresh Benchmark | https://lup.lub.lu.se/student-papers/record/9052567/file/9052584.pdf | 2024 |
| 28 | StackOverflow - Featuretools Target Leakage | https://stackoverflow.com/questions/50639687/should-we-exclude-target-variable-from-dfs-in-featuretools | 2025 |
| 29 | TSFresh Efficient Feature Extraction | https://converter.brightcoding.dev/blog/tsfresh-the-powerful-python-tool-automating-time-series-features | 2026 |
| 30 | GeeksforGeeks - TSFresh Tutorial | https://www.geeksforgeeks.org/data-analysis/creating-powerful-time-series-features-with-tsfresh/ | 2024 |

---

## Appendix A: Feature Count Estimates

### With Our Multi-Table Structure

Assuming:
- orders table: 20 columns
- order_lines table: 15 columns
- operations table: 15 columns
- ~60 days of data
- 1000-5000 orders

**Featuretools output (max_depth=2):**
- Aggregation primitives: ~15 agg x ~10 numeric cols x 2 child tables = ~300
- Transform primitives: ~10 transforms x ~20 cols = ~200
- Stacked (depth-2): ~50-100
- **Total: ~550-600 features** (before selection)

**TSFresh EfficientFCParameters:**
- ~200 features per time series column
- 5 key columns = ~1000 features
- After select_features with fdr_level=0.01: ~50-100 retained

**After Boruta + RFE:**
- Starting: ~600 (Featuretools) + ~100 (TSFresh) + ~20 (manual) = ~720
- After Boruta: ~100-200
- After RFE: **10-20 final features**

### Recommended Final Feature Count

With ~60 days of data (60 * n_series samples), target **10-20 features** to avoid overfitting. This follows the rule of thumb: `n_features <= n_samples / 30` for regularized models.
