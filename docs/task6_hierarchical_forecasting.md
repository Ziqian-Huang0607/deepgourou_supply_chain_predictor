# Hierarchical Forecasting & Temporal Aggregation: Research Report

## Executive Summary

For **90 days of daily training data**, the optimal forecasting strategy depends on the data characteristics and business requirements. Based on extensive research into academic literature, the M5 competition, and industry best practices, here are the key findings:

| Scenario | Recommended Approach |
|----------|---------------------|
| **No cross-sectional hierarchy, just daily forecasts** | **Predict daily directly** with LightGBM/XGBoost using temporal features |
| **Need both monthly AND daily forecasts** | **Temporal hierarchy**: predict at multiple frequencies, reconcile with MinT or bottom-up |
| **Intermittent/sparse daily demand (many zeros)** | **ADIDA/MAP approach**: aggregate to weekly → forecast → disaggregate to daily |
| **Need aggregate totals + breakdown** | **Bottom-up**: predict at lowest level, aggregate up |
| **Top-level accuracy is most important** | **Predict monthly directly + disaggregate using historical proportions** |

**Bottom line**: For 90 days of training data, **do NOT aggregate to monthly first and then distribute to daily** unless your data is highly intermittent (many zero-demand days). The loss of granularity and the difficulty of disaggregation typically outweigh any noise-reduction benefits. Instead, **forecast daily directly** using a tree-based model (LightGBM/XGBoost) with rich temporal features, and if hierarchy consistency is needed, apply reconciliation post-hoc.

---

## Table of Contents

1. [When to Aggregate vs. Forecast at Native Frequency](#1-when-to-aggregate-vs-forecast-at-native-frequency)
2. [Reconciliation Methods: Bottom-Up, Top-Down, Middle-Out, MinT](#2-reconciliation-methods)
3. [M5 Competition: Key Lessons](#3-m5-competition-key-lessons)
4. [Temporal Hierarchies (Athanasopoulos et al. 2017)](#4-temporal-hierarchies)
5. [MinT/MAP Reconciliation](#5-mintmap-reconciliation)
6. [ADIDA and Intermittent Demand](#6-adida-and-intermittent-demand)
7. [Recommendation for 90 Days of Training Data](#7-recommendation-for-90-days-of-training-data)
8. [Implementation Guide](#8-implementation-guide)
9. [Key References](#9-key-references)

---

## 1. When to Aggregate vs. Forecast at Native Frequency

### 1.1 When Temporal Aggregation HELPS

Temporal aggregation (forecasting at a higher level, then disaggregating) provides the strongest benefits when (MetricGate, 2024; Kourentzes et al., 2017):

| Condition | Why Aggregation Helps |
|-----------|----------------------|
| **High-frequency noise** (measurement error, daily shocks) | Aggregation smooths out noise |
| **Strong low-frequency structure** (slow trends, macro cycles) | Monthly models capture trends more stably |
| **Long forecast horizon** (>30 days ahead) | Daily model's seasonality assumptions get fragile |
| **Intermittent demand** (many zero-demand days) | Aggregation converts near-zero noise into countable Poisson demands |
| **Very short history relative to seasonality** | Aggregation provides more complete seasonal cycles |

### 1.2 When Aggregation HURTS

Aggregation can **damage** forecast accuracy when:

| Condition | Why Aggregation Hurts |
|-----------|----------------------|
| **Short-run signals matter** (sharp turning points, intra-week patterns) | Aggregation averages away important daily patterns |
| **Day-of-week effects are strong** | Monthly aggregation loses weekly seasonality |
| **Short forecast horizon** (next 7-14 days) | Daily models capture recent dynamics better |
| **Need operational daily decisions** | Disaggregation introduces additional error |
| **Non-intermittent, stable demand** | Information loss from aggregation exceeds noise reduction |

### 1.3 Key Finding: Bottom-Up Often Beats Aggregation

A comprehensive study using M4 competition data found that **Bottom-Up (BU) forecasting—forecasting at the native frequency and aggregating—is more accurate than temporal aggregation for ~50% of monthly/quarterly series and up to 75% of daily series** (Petropoulos & Kourentzes, 2015). This was described as "surprising" and challenges the prevalent thinking that temporal aggregation is always beneficial for lead-time forecasting.

### 1.4 Practical Guidance on Aggregation Level

From the research (Athanasopoulos et al., 2017; Kourentzes et al., 2017):

- **For noisy, intermittent, or long-horizon series**: Aggregation provides bigger wins
- **For smooth, well-behaved series**: Single-frequency forecasting is sufficient; aggregation gains are marginal
- **The safest move**: Combine rather than replace—keep the base-frequency forecast and combine with aggregated forecasts using inverse-variance weights
- **For 90 days of daily data**: You have ~12 weeks of data or ~3 months. This is **borderline** for monthly aggregation—you have only 3 monthly observations which is very few.

---

## 2. Reconciliation Methods

### 2.1 Overview of Methods

| Method | How It Works | Strength | Weakness | Best For |
|--------|-------------|----------|----------|----------|
| **Bottom-Up (BU)** | Forecast at lowest level, sum upward | Captures local detail, no disaggregation error | Noise amplification at low levels, ignores top-level signal | Stable demand, rich history at bottom level |
| **Top-Down (TD)** | Forecast at top level, allocate downward using proportions | Stable totals, simple, captures macro trends | Loses local patterns, allocation may be inaccurate | New stores, sparse SKU data, strategic planning |
| **Middle-Out** | Forecast at middle level, split both directions | Balances both ends | Requires choosing the "middle" level | Mixed-maturity portfolios |
| **MinT (Minimum Trace)** | Statistically optimal combination of all levels | Mathematically optimal, borrows strength across all levels | Requires estimating covariance matrix; computationally expensive | Mature data, complex hierarchies |
| **OLS Reconciliation** | MinT with W=I (identity matrix) | Simple, fast, no covariance estimation | Ignores different reliabilities across series | Quick baseline, limited data |
| **WLS (Structural Scaling)** | MinT with diagonal W | Good balance of accuracy and simplicity | Ignores cross-correlations | Production systems, short series |

### 2.2 Which Reconciliation Method is Best?

The evidence strongly favors **MinT** when computationally feasible (Wickramasuriya et al., 2019):

- MinT minimizes the expected mean squared error of reconciled forecasts
- It is optimal with respect to both point forecast accuracy and (for Gaussian base forecasts) the logarithmic score
- In practice, MinT often produces forecasts that are more accurate than ANY individual level's base forecast—it "borrows strength" across levels

For **short time series** (like 90 days), the research shows:
- MinT is **remarkably robust** even with very few observations (as few as 15-30)
- When covariance estimation is difficult, **WLS with structural scaling** is a good alternative
- **Iterative MinT (MinTit)** can further improve accuracy for very short series by reconciling sub-hierarchies iteratively

### 2.3 Practical Comparison

```
Forecast Accuracy Ranking (general case):
1. MinT-shrinkage (best overall)
2. WLS-structural / MinTit (good for short series)
3. Bottom-Up (simple, often competitive)
4. OLS (fast baseline)
5. Top-Down (useful when bottom data is very sparse)
```

---

## 3. M5 Competition: Key Lessons

### 3.1 Background

The M5 Forecasting Competition (2020) was the largest hierarchical forecasting competition to date:
- **42,840 hierarchical time series** from Walmart
- **12 aggregation levels** (from total → state → store → category → department → product-store)
- **1,941 days** of history (~5.3 years)
- Forecast horizon: **28 days ahead**
- Evaluation: WRMSSE (Weighted Root Mean Squared Scaled Error) across all levels

### 3.2 Key Findings from M5

**Finding 1: ML (especially LightGBM) dominated**
- M5 was the first M competition where ALL top-50 methods were ML-based
- LightGBM achieved >14% improvement over best statistical benchmarks
- Top 5 methods achieved >20% improvement
- 4 of top 5 solutions used LightGBM

**Finding 2: Simple ensembles and feature engineering mattered most**
- The winning solution used LightGBM with grouped data and equal-weight ensemble
- Feature engineering (lags, rolling statistics, calendar features, prices) was crucial
- Cross-validation strategy was the single most important methodological choice

**Finding 3: Hierarchy handling varied**
- **2nd place solution**: Used hierarchical alignment—N-BEATS for top levels, LightGBM for bottom level, with a tuning parameter to align bottom forecasts to top-level forecasts
- **3rd place solution**: Did NOT exploit hierarchy explicitly—relied on model diversity and validation strategy
- **The "magic multiplier"**: Many top teams found that post-hoc scaling by ~0.97 improved results

**Finding 4: Bottom-up outperformed 92% of entrants**
- Simple exponential smoothing with bottom-up aggregation outperformed 92% of competition entrants
- This highlights that sophisticated reconciliation is not always necessary

### 3.3 What M5 Tells Us About 90-Day Forecasting

M5 had ~5 years of data. For a **90-day** scenario, the key insights are:
- **Tree-based models** (LightGBM/XGBoost) are the strongest baseline
- **Cross-learning** (training one model on all series) is highly effective
- **Feature engineering matters more than reconciliation method**
- **Bottom-up is a strong, simple baseline** that's hard to beat
- Most top solutions did NOT use sophisticated reconciliation—hierarchy was handled implicitly

---

## 4. Temporal Hierarchies (Athanasopoulos et al. 2017)

### 4.1 The Core Idea

Athanasopoulos et al. (2017) introduced the framework of **temporal hierarchies** in their seminal paper "Forecasting with temporal hierarchies" (European Journal of Operational Research, 262(1), 60-74).

**Key insight**: A time series observed at a given frequency can be aggregated to create multiple related time series at different frequencies. For daily data:

```
Daily (level 0) → Weekly (level 1) → Bi-weekly (level 2) → Monthly (level 3) → Quarterly (level 4)
```

These form a natural hierarchy where forecasts at different levels should be **coherent** (e.g., the sum of 7 daily forecasts should equal the weekly forecast).

### 4.2 Why Temporal Hierarchies Work

1. **Different frequencies reveal different signals**: Daily shows day-of-week patterns; monthly shows trends; quarterly shows macro cycles
2. **Combination improves accuracy**: Combining forecasts from multiple frequencies is like ensemble learning—it averages out model-specific errors
3. **Reconciliation ensures coherence**: Business decisions made at different time granularities use consistent forecasts

### 4.3 The Optimal Reconciliation Formula

For temporal hierarchies, the reconciled forecast is computed as:

```
ỹ = S * (S' * W^(-1) * S)^(-1) * S' * W^(-1) * ŷ
```

Where:
- `ŷ` = vector of base forecasts at ALL temporal levels
- `S` = summing matrix (encodes aggregation structure)
- `W` = covariance matrix of base forecast errors
- `ỹ` = reconciled coherent forecasts

### 4.4 Key Result: Bottom-Up is Optimal for Aggregated ARIMA Models

A 2024 theoretical paper proved an important result: **within the framework of temporally aggregated ARIMA models, the optimal forecast reconciliation technique is the bottom-up approach** (Lemma and Theorem 1, "Effective Forecasting in Temporal Hierarchies", 2024).

This provides theoretical justification for why bottom-up often works well in practice.

### 4.5 Practical Implementation of Temporal Hierarchies

For **90 days of daily data**, the temporal hierarchy would be:
- **Daily**: 90 observations (original)
- **Weekly**: ~12-13 observations (sparse!)
- **Monthly**: ~3 observations (too few for reliable forecasting)

**Verdict**: With only 90 days, weekly and especially monthly aggregation levels have too few observations for reliable model fitting. Temporal hierarchy reconciliation is **not recommended** for such short series unless you have strong seasonal patterns.

---

## 5. MinT/MAP Reconciliation

### 5.1 Minimum Trace (MinT) Reconciliation

Proposed by Wickramasuriya, Athanasopoulos, and Hyndman (2019) in "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization" (JASA, 114(526), 804-819).

**Core idea**: Find the reconciled forecast that minimizes the trace of the forecast error covariance matrix.

**The reconciliation matrix P:**
```
P = (S' * W^(-1) * S)^(-1) * S' * W^(-1)
```

**Variants based on W estimation:**

| Variant | W Assumption | Use Case |
|---------|-------------|----------|
| **MinT-ols** | W = I (identity) | Fast, no estimation needed |
| **MinT-wls** | W = diagonal(variances) | Good balance, ignores correlations |
| **MinT-shrink** | W = shrunk covariance | Best accuracy, handles "series > observations" |
| **MinT-sam** | W = sample covariance | Unstable when p > T |

### 5.2 Practical Considerations for MinT

**When MinT works well:**
- Sufficient history to estimate W reliably (ideally T > 2p where p = number of series)
- Hierarchy has meaningful cross-sectional structure
- Base forecasts at different levels provide complementary information

**When MinT struggles:**
- Very short time series (W estimation becomes unstable)
- Very large hierarchies (inversion of large matrices)
- Highly correlated base forecast errors

**For 90 days of data:**
- If your hierarchy has <10 series: MinT-shrink is feasible and recommended
- If your hierarchy has 10-50 series: Use WLS (structural scaling) instead
- If your hierarchy has >50 series: Bottom-up or OLS reconciliation may be more stable

### 5.3 Iterative MinT (MinTit) for Short Series

A 2024 paper introduced iterative versions of MinT that are specifically designed for very short time series:

**Algorithm**: Instead of estimating the full covariance matrix, iteratively reconcile sub-hierarchies (e.g., parent + children pairs) until convergence.

**Results**: MinTit achieved higher error reduction than standard MinT for ETS and ARIMA base forecasts, especially for series with T=15-30 observations. For T=60, standard MinT performed better.

**Relevance for 90 days**: With 90 daily observations, MinTit is worth considering if you have a moderate-sized hierarchy.

---

## 6. ADIDA and Intermittent Demand

### 6.1 The ADIDA Framework

**ADIDA** (Aggregate-Disaggregate Intermittent Demand Approach), proposed by Nikolopoulos et al. (2011):

1. **Aggregate**: Sum demand into larger time buckets (e.g., daily → weekly)
2. **Forecast**: Apply any forecasting method to the aggregated series
3. **Disaggregate**: Break the forecast back into original time units

**Key insight**: Aggregation reduces or eliminates intermittence (zeros), enabling the use of standard forecasting methods instead of specialized intermittent demand methods like Croston's.

### 6.2 Optimal Aggregation Level

Research shows there exists an **optimal aggregation level unique to each series**. For M5 data:
- A 14-day aggregation window minimized mean RMSSE across all series
- However, optimal aggregation varies significantly by series
- Optimizing per-series (rather than using a fixed window) improved WRMSSE from 0.765 to 0.685

### 6.3 When to Use ADIDA for 90 Days of Data

| Condition | ADIDA Recommended? | Optimal Aggregation |
|-----------|-------------------|---------------------|
| >50% zero-demand days | **Yes** | 7-14 days (weekly or bi-weekly) |
| 20-50% zero-demand days | Maybe | 3-7 days |
| <20% zero-demand days | No | Forecast daily directly |
| Strong day-of-week pattern | Maybe | 7 days (preserves weekly structure) |

### 6.4 Disaggregation Methods

Once you have an aggregated forecast, how do you disaggregate?

| Method | Description | Best For |
|--------|-------------|----------|
| **Equal weights** | Divide equally (e.g., weekly/7) | No strong within-period pattern |
| **Historical proportions** | Use historical within-period profile | Stable day-of-week patterns |
| **Forecast proportions** | Use forecasted proportions | Changing patterns |
| **Seasonal indices** | Use pre-computed seasonal factors | Strong seasonal patterns |

---

## 7. Recommendation for 90 Days of Training Data

### 7.1 Decision Framework

```
                    START: 90 days of daily data
                            |
                Is there a cross-sectional hierarchy?
                            |
                    YES              NO
                     |                |
            How many bottom series?    Is demand intermittent?
             |        |        |            |
           <10      10-50     >50        YES      NO
            |        |        |            |        |
          MinT-   WLS or    Bottom-    ADIDA    LightGBM/
          shrink  MinTit      up       (weekly   XGBoost
                                      → daily)   daily direct
```

### 7.2 Specific Recommendations

**Scenario A: No hierarchy, just daily forecasts (single series)**
- **Approach**: Forecast daily directly using LightGBM or XGBoost
- **Features**: Lags (1-28 days), rolling means/std (7, 14, 28 days), calendar features, trends
- **Why**: 90 daily observations is sufficient for tree-based models with rich features
- **Do NOT**: Aggregate to monthly first—you only have 3 months, which is too few

**Scenario B: Need both monthly totals AND daily breakdown**
- **Approach**: Temporal hierarchy with reconciliation
- **Daily forecasts**: LightGBM at daily level
- **Weekly/Monthly**: Aggregate daily forecasts → reconcile
- **Reconciliation**: Bottom-up (sum daily → weekly → monthly) or MinT-shrink if <10 series
- **Why**: Bottom-up is theoretically optimal for ARIMA models and empirically strong

**Scenario C: Highly intermittent daily demand (many zeros)**
- **Approach**: ADIDA
- **Aggregate** to weekly (12-13 observations is workable)
- **Forecast** weekly using LightGBM
- **Disaggregate** using historical day-of-week proportions
- **Why**: Aggregation removes intermittence; 12-13 weekly points is better than 90 daily with 50%+ zeros

**Scenario D: Need forecasts at multiple cross-sectional levels**
- **Approach**: Cross-sectional hierarchical forecasting
- **Bottom level**: LightGBM for daily forecasts
- **Reconciliation**: Bottom-up if >50 bottom series; MinT-shrink if <10 series
- **Why**: M5 showed bottom-up + LightGBM is very strong; MinT for smaller hierarchies

### 7.3 Why NOT "Monthly-then-Distribute" for 90 Days?

The "predict monthly, then disaggregate to daily" approach has critical flaws with only 90 days:

| Problem | Explanation |
|---------|-------------|
| **Only 3 monthly observations** | Too few for any model to learn patterns reliably |
| **Disaggregation error** | Even if monthly forecast is perfect, splitting it introduces additional error |
| **Loses day-of-week patterns** | Monthly model cannot capture weekly seasonality |
| **Information loss** | Aggregation discards the rich daily patterns that 90 days provides |

**The exception**: If your data is extremely intermittent (>50% zeros), aggregating to weekly (not monthly) first makes sense because the daily signal is too noisy.

---

## 8. Implementation Guide

### 8.1 Python Libraries for Hierarchical Forecasting

| Library | Reconciliation Methods | Best For |
|---------|----------------------|----------|
| **`hierarchicalforecast`** (Nixtla) | BottomUp, TopDown, MiddleOut, MinT (ols, wls, shrink) | General use, well-maintained |
| **`bayesReconPy`** | Conditioning and projection methods, discrete reconciliation | Probabilistic reconciliation |
| **`scikit-hts`** | BU, TD, MiddleOut, MinT (various) | scikit-learn integration |
| **`hts`** (R) | Comprehensive reconciliation suite | R users |
| **`FoReco`** (R) | Cross-temporal reconciliation | Advanced temporal+spatial reconciliation |

### 8.2 Code Example: Hierarchical Forecasting with HierarchicalForecast

```python
# pip install hierarchicalforecast statsforecast datasetsforecast

import pandas as pd
import numpy as np
from statsforecast.core import StatsForecast
from statsforecast.models import LightGBM, XGBoost
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace

# 1. Prepare data with hierarchical structure
# Y_df columns: ['unique_id', 'ds', 'y']
# S_df: summing matrix (defines hierarchy)
# tags: dictionary mapping level names to series ids

# 2. Generate base forecasts at all levels
fcst = StatsForecast(
    models=[LightGBM(), XGBoost()],
    freq='D',
    n_jobs=-1
)
Y_hat_df = fcst.forecast(df=Y_train_df, h=horizon)

# 3. Reconcile forecasts
reconcilers = [
    BottomUp(),
    TopDown(method='forecast_proportions'),
    MinTrace(method='mint_shrink'),  # Best general-purpose
    MinTrace(method='ols'),          # Fast baseline
]

hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(
    Y_hat_df=Y_hat_df,
    Y_df=Y_train_df,
    S_df=S_df,
    tags=tags
)

# 4. Evaluate
# Y_rec_df now contains reconciled forecasts for all hierarchy levels
```

### 8.3 Code Example: ADIDA for Intermittent Demand

```python
import pandas as pd
import numpy as np
from statsforecast.models import LightGBM

# 1. Aggregate daily to weekly
def aggregate_to_weekly(df):
    """Aggregate daily demand to weekly."""
    df = df.copy()
    df['week'] = df['ds'].dt.to_period('W')
    weekly = df.groupby('week')['y'].sum().reset_index()
    weekly['ds'] = weekly['week'].dt.to_timestamp()
    return weekly[['ds', 'y']]

# 2. Forecast weekly
def forecast_weekly(weekly_df, horizon_weeks=4):
    """Forecast weekly aggregated demand."""
    # Use LightGBM or any model
    model = LightGBM()
    model.fit(weekly_df['y'].values)
    forecasts = model.predict(h=horizon_weeks)
    return forecasts

# 3. Disaggregate using historical day-of-week proportions
def disaggregate_weekly(weekly_forecast, historical_df):
    """Break weekly forecast into daily using day-of-week profile."""
    # Calculate day-of-week proportions from history
    historical_df['dow'] = historical_df['ds'].dt.dayofweek
    dow_profile = historical_df.groupby('dow')['y'].mean()
    dow_proportions = dow_profile / dow_profile.sum()

    # Apply proportions to weekly forecast
    daily_forecasts = weekly_forecast * dow_proportions.values
    return daily_forecasts
```

### 8.4 Feature Engineering for Daily Forecasting (90 Days)

With only 90 days, feature engineering is critical. Key features from M5 top solutions:

| Feature Category | Specific Features |
|-----------------|-------------------|
| **Lag features** | y(t-1), y(t-7), y(t-14), y(t-21), y(t-28), y(t-364) |
| **Rolling statistics** | mean/std/min/max of last 7, 14, 28, 60 days |
| **Calendar features** | day_of_week, month, day_of_month, is_weekend, is_month_start/end |
| **Trend** | linear trend, days_since_start |
| **External** | price, promotions, holidays, SNAP/foodstamp indicators |

### 8.5 Cross-Validation Strategy

With 90 days, use **time-series cross-validation**:

```
Fold 1: Train [day 1-60], Validate [day 61-75], Test [day 76-90]
Fold 2: Train [day 1-75], Validate [day 76-90]
Fold 3: Train [day 1-30], Validate [day 31-45]  # Stability check
```

Key insight from M5: **The cross-validation strategy was more important than model choice.**

---

## 9. Key References

### Foundational Papers

1. **Athanasopoulos, G., Hyndman, R.J., Kourentzes, N., & Petropoulos, F. (2017)**. "Forecasting with temporal hierarchies." *European Journal of Operational Research*, 262(1), 60-74. [Seminal paper on temporal hierarchies]

2. **Wickramasuriya, S.L., Athanasopoulos, G., & Hyndman, R.J. (2019)**. "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization." *Journal of the American Statistical Association*, 114(526), 804-819. [MinT reconciliation]

3. **Kourentzes, N. & Athanasopoulos, G. (2019)**. "Cross-temporal coherent forecasts for Australian tourism." *Annals of Tourism Research*, 75, 393-409. [Cross-temporal reconciliation]

4. **Athanasopoulos, G., Hyndman, R.J., Kourentzes, N., & Panagiotelis, A. (2024)**. "Forecast reconciliation: A review." *International Journal of Forecasting*, 40(2), 430-456. [Comprehensive review]

### M5 Competition Papers

5. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022)**. "M5 accuracy competition: Results, findings, and conclusions." *International Journal of Forecasting*, 38(4), 1346-1364.

6. **Anderer, M. & Li, F. (2022)**. "Hierarchical forecasting with a top-down alignment of independent level forecasts." *International Journal of Forecasting*. [2nd place M5 solution]

7. **Makridakis, S., et al. (2022)**. "The M5 uncertainty competition: Results, findings and conclusions." *International Journal of Forecasting*, 38(4), 1365-1385.

### Temporal Aggregation & ADIDA

8. **Nikolopoulos, K., et al. (2011)**. "An aggregate-disaggregate intermittent demand approach (ADIDA) to forecasting: an empirical proposition and analysis." *Journal of the Operational Research Society*, 62, 1695-1704.

9. **Kourentzes, N., Petropoulos, F., & Trapero, J.R. (2014)**. "Improving forecasting by estimating time series structural components across multiple frequencies." *International Journal of Forecasting*, 30(2), 291-302.

10. **Petropoulos, F. & Kourentzes, N. (2015)**. "Forecast combinations for intermittent demand." *Journal of the Operational Research Society*, 66(6), 914-924.

### Recent Advances

11. **"Effective Forecasting in Temporal Hierarchies" (2024)**. arXiv:2407.02367. [Proves bottom-up is optimal for aggregated ARIMA models]

12. **"Iterative Trace Minimization for the Reconciliation of Very Short Hierarchical Time Series" (2024)**. arXiv:2409.18550. [MinTit for short series]

13. **"Insights into regression-based cross-temporal forecast reconciliation" (2024)**. arXiv:2410.19407. [Sequential vs optimal reconciliation]

---

## Summary Table: Decision Matrix for 90 Days of Daily Data

| Question | If YES | If NO |
|----------|--------|-------|
| Is demand intermittent (>30% zeros)? | Use ADIDA (aggregate to weekly, forecast, disaggregate) | Forecast daily directly |
| Is there a cross-sectional hierarchy? | Use hierarchical reconciliation (BottomUp or MinT) | Single-series model |
| Do you need monthly totals too? | Build temporal hierarchy + reconcile | Focus on daily only |
| Is top-level accuracy critical? | Top-down alignment (like M5 2nd place) | Bottom-up |
| Is the hierarchy large (>50 series)? | Bottom-up or WLS (not full MinT) | MinT-shrink is feasible |
| Do you need probabilistic forecasts? | bayesReconPy for probabilistic reconciliation | Point forecasts with HierarchicalForecast |

---

## Final Verdict

> **For 90 days of daily training data, forecast DAILY directly using LightGBM or XGBoost with rich temporal features.** Do NOT aggregate to monthly first—you only have 3 monthly observations. Only aggregate to weekly if demand is highly intermittent (>30% zeros). If hierarchical consistency is needed across multiple levels, apply bottom-up reconciliation or MinT (with shrinkage for <10 series, WLS for larger hierarchies).
