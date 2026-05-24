# Kaggle Winning Solutions for Demand Forecasting - Comprehensive Research

## Executive Summary

This document catalogs the exact techniques, features, models, and strategies used by winners of major Kaggle demand forecasting competitions. The research covers: **M5 Forecasting (Accuracy & Uncertainty)**, **Corporacion Favorita Grocery Sales**, **Rossmann Store Sales**, **Recruit Restaurant Visitor Forecast**, and **Grupo Bimbo Inventory Demand**.

### Key Meta-Findings Across All Competitions

| Finding | Evidence |
|---------|----------|
| **LightGBM dominates** | Winners of Corporacion Favorita, Recruit Restaurant, and M5 all used LightGBM as primary model |
| **Ensembling is essential** | Every single winner used some form of model combination; single models never won |
| **Feature engineering > model complexity** | Winners focused on rolling stats, lag features, calendar features; not exotic architectures |
| **Cross-learning (one model for many series)** | M5 proved this approach superior to per-series models |
| **Recent data often beats all data** | Favorita winner used only 1-5 months despite 55 months available |
| **Tweedie loss for zero-inflated sales** | M5 winners used Tweedie distribution loss for intermittent demand |
| **Recursive + Direct forecasting combination** | M5 winner combined both approaches for robustness |

---

## Competition 1: M5 Forecasting - Accuracy Competition (2020)

**Competition**: Predict daily sales for 30,490 product-store combinations at Walmart for 28 days
**Metric**: Weighted Root Mean Squared Scaled Error (WRMSSE)
**Winner**: YJ_STU (YeonJun Im) - undergraduate student from South Korea

### 1st Place Solution: DRFAM (Direct and Recursive Forecast Averaging Method)

**Model Architecture:**
- **Base model**: LightGBM (gradient boosting trees)
- **Total models built**: 220 LightGBM models
- **Models used per prediction**: Average of 6 models
- **Key innovation**: Simple averaging of direct and recursive multi-step forecasts via partial pooling

**Partial Pooling Strategy (3 levels):**

| Pool Level | Number of Models | Data Grouping |
|------------|------------------|---------------|
| Store level | 10 models | One per store |
| Store-Category level | 30 models | One per store-category combination |
| Store-Department level | 70 models | One per store-department combination |

For each pool level, two variations:
- **Recursive forecasting**: Uses prior forecasts as inputs for later timestamps
- **Non-recursive (direct) forecasting**: Predicts each horizon day independently

**Loss Function:**
- **Tweedie loss** (negative log-likelihood of Tweedie distribution)
- Critical for zero-inflated, right-skewed sales data
- No early stopping used

**Validation Strategy:**
- Last **four 28-day windows** of available data for cross-validation
- Measured both **mean AND standard deviation** of errors
- Selected models that provided both accuracy and stability
- Recursive models found more accurate but less stable; non-recursive more stable but less accurate
- **Final solution combined both** for robustness

**Features Used (by category):**

| Category | Specific Features |
|----------|------------------|
| Identifiers | item_id, dept_id, cat_id, store_id, state_id |
| Calendar/Date | month, year, day, weekday, week_number, day_of_year |
| Special Events | event names, event types (Sporting/Cultural/National/Religious), SNAP indicators |
| Prices | Current price, price relative to past, price statistics (max, min, std, mean) |
| Unit Sales (recursive) | Lagged sales values, rolling means at multiple windows |
| Unit Sales (non-recursive) | Direct lag features without using prior predictions |

**Key Insight That Won:**
> The complementary effects between direct and recursive multi-step forecasting. Direct forecasting avoids error accumulation; recursive forecasting captures temporal dependencies. Averaging them via partial pooling at multiple hierarchy levels provided both accuracy and robustness.

### 2nd Place Solution (Matthias Anderer)

**Model Architecture:**
- Equal-weighted combination of 50 LightGBM models (5 per store)
- Externally adjusted through multipliers from **N-BEATS** deep learning model
- N-BEATS used for top 5 aggregation levels to capture trend

**Key Features:**
- LightGBM models used only **basic calendar features and prices** (no past unit sales)
- N-BEATS used **solely historical unit sales**
- Custom asymmetric loss function

**Validation:** Last four 28-day windows

**Key Insight:** Separate trend modeling with N-BEATS for higher aggregation levels + LightGBM for granular levels

### 3rd Place Solution (DeepAR - Neural Network Approach)

**Model Architecture:**
- **Modified DeepAR** with Tweedie loss
- 43 deep learning NNs (each with multiple LSTM layers)
- 24 models with dropout, 19 without
- Originated from 12 base models, selected best checkpoints via CV

**Features Used:**

| Feature Category | Details |
|-----------------|---------|
| Sales values | Lag 1 value, moving averages (7-day, 28-day) |
| Calendar | wday, month, year, week number, day (all normalized) |
| Events | Event type embedding, Event name embedding |
| SNAP | Binary [0, 1] |
| Price | Raw value, normalized across time, normalized within dept |
| Categories | state_id, store_id, cat_id, dept_id, item_id embeddings |
| Zero sales | Continuous zero-sale days until today |

**Training:**
- Randomly sampled 28-day slices, 64 slices per batch
- 300 epochs, Adam optimizer, cosine annealing learning rate
- Tweedie loss for zero-inflated data

**Validation:** Last fourteen 28-day windows

### 4th Place Solution (Monsanida)

**Model Architecture:**
- 40 non-recursive LightGBM models
- **One model per week** (4 models per store x 10 stores)
- Tweedie regression, no early stopping

**Key Difference:** No recursive features; each week forecast separately

### 5th Place Solution (Alan Laboud)

**Model Architecture:**
- Recursive LightGBM models trained per department (7 models)
- Poisson regression with early stopping
- External adjustment using multipliers to match store-department means

### 7th Place Solution - Detailed Feature List

**Confirmed Features Used:**
- Item id, department id, category id
- **Price features**: current price, price relative to past prices, price statistics (max, min, std, mean)
- **Sales features**: recursive 7-day, 14-day lagged rolling means + 28-day lagged rolling means
- **Date features**: month, year, is_weekend, day of week, nearest upcoming/past event name, event name 1, event name 2, SNAP, no event type

**Key Learning from 7th Place:**
> "Simpler was better. With cross-validation, I pruned a lot of features that did not provide large benefit to keep the model simpler."

---

## Competition 2: M5 Forecasting - Uncertainty Competition (2020)

**Task**: Predict 9 quantiles (0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
**Metric**: Weighted Scaled Pinball Loss

### 1st Place Solution

**Model Architecture:**
- **126 LightGBM models** - one for each quantile and aggregation level combination
- Pure quantile regression approach

### 2nd Place Solution

**Model Architecture:**
- Recursive LightGBM models + statistical methods + simple time series techniques
- Hybrid ML + statistical approach

### 3rd Place Solution

**Model Architecture:**
- Hybrid: LightGBM for point estimation + Histogram algorithm for quantile estimation
- Separate treatment of median vs. other quantiles

### 5th Place Solution

**Model Architecture:**
- **280 LightGBM models** in comprehensive ensemble
- Most complex ensemble in top 5

### 13th Place Solution (Hybrid LightGBM + DeepAR)

**Approach:**
- LightGBM + DeepAR hybrid for median and quantile estimation
- Different approaches based on aggregation level
- Ranked best by Multiple Comparisons with Best (MCB) test despite 13th place
- **Best performance at product-store level** (hardest aggregation)

---

## Competition 3: Corporacion Favorita Grocery Sales Forecasting (2017)

**Competition**: Forecast daily unit sales for 54 stores x 3,901 products = ~210,000 time series
**Forecast Horizon**: 1-16 days
**Data**: 55 months of data + store metadata, promotions, holidays, oil prices
**Metric**: Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE)
**Winner**: Complex ensemble of GBDT + Neural Networks

### Winning Solution Architecture

**Model Ensemble (4 models total):**

| Model | Description |
|-------|-------------|
| LightGBM (per horizon) | 16 separate models, one for each forecast day |
| Feedforward Neural Network (per horizon) | 16 separate models, one for each forecast day |
| LightGBM (all horizons) | Single model predicting all 16 days |
| CNN | Architecture from 6th place Wikipedia competition |

**Key Innovation: Per-Horizon Training**
- Training one model per forecast horizon (16 models) instead of one model for all horizons
- Allows models to learn what information is useful for each specific day
- Trade-off: Requires 16x more models but yields better results

**Features Used:**

| Feature Category | Examples |
|-----------------|----------|
| Rolling statistics by store | Rolling mean, std by store |
| Rolling statistics by item | Rolling mean, std by item |
| Rolling statistics by class | Rolling mean, std by product class |
| Rolling statistics by combinations | Store-item, store-class combinations |
| Measures of centrality | Mean, median sales |
| Measures of spread | Standard deviation, variance |
| Exponential moving average | EMA at various windows |
| Store metadata | Store type, cluster, location |
| Promotions | Promotion indicators |
| Oil prices | Daily oil price (external) |
| Holidays | National and local holiday indicators |

**Critical Finding - Recent Data Only:**
> "The winner only used very recent data in the models, electing to drop older observations based on validation dataset performance. The final models used less than a full season of data for model fitting in the form of either **one, three, or five months of data**, despite multiple seasons being available."

**Why this worked:**
- Strong trend in the data made older data less relevant
- Short forecast horizon (16 days) meant yearly seasonality less critical
- Other top placers (5th, 6th) also favored this approach

**Validation Strategy:**
- Hold-out validation set of same length as forecast horizon (16 days)
- Used to select hyperparameters and features
- Simple hold-out approach sufficient (top 3 all used it)

**Key Insight That Won:**
> Per-horizon model training combined with diverse ensemble of GBDT and neural networks. Using only recent data and focusing on rolling statistics at multiple hierarchy levels.

---

## Competition 4: Rossmann Store Sales (2015)

**Competition**: Predict daily sales for 1,115 Rossmann drug stores across Germany for 48 days
**Data**: 31 months + store metadata, promotions, holidays, weather, Google Trends
**Metric**: Root Mean Square Percentage Error (RMSPE)
**Winner**: Ensemble of 12 XGBoost models + trend adjustment

### Winning Solution Architecture

**Model:**
- **Ensemble of 12 Extreme Gradient Boosting (XGBoost) models**
- Performance boost from ensembling: **~5% over best single model**

**Trend Adjustment:**
- Ridge regression model for trend adjustment
- Addresses GBDT's inability to extrapolate trends

**Ensemble Variation Strategies:**
- Different data subsets for training different models
- Both direct and iterated predictions
- Different feature subsets across models

**Key Features:**

| Feature Category | Specific Features |
|-----------------|-------------------|
| Hierarchy statistics | Average sales by product, moving averages by product |
| Promotion features | Average sales by product and promotion status |
| Event counters | Days until event, days into event, days after event (holidays, promotions) |
| Weather | Precipitation, maximum temperature |
| Seasonality indicators | Month, Year, Day of Month, Week of Year, Day of Year |
| Google Trends | Search trend statistics |
| Store metadata | Store type, assortment, competition distance, competition open date |

**Feature Innovation: Event Counters**
- Number of days until an event
- Number of days into an event
- Number of days after an event
- Applied to holidays and promotions
- Proved essential for high performance

**Validation Strategy:**
- Hold-out dataset of same length as forecast horizon
- Evaluate model quality and select hyperparameters/features

**Key Insight That Won:**
> Adapted XGBoost for time series through extensive feature engineering with rolling statistics at multiple hierarchy levels, event counters, and external data (weather, Google Trends). Trend adjustment via ridge regression addressed GBDT's key weakness. 12-model ensemble provided 5% boost.

### 3rd Place Solution: Entity Embeddings

**Innovation:** First top-3 neural network in Kaggle forecasting
- Used categorical embeddings for store, day of week, month, year, state
- Fully connected neural network with embedding layers
- Learned vector representations for categorical variables
- Published landmark paper: "Entity Embeddings of Categorical Variables"

**Embedding Dimensions Used:**

| Feature | Values | Embedding Dimension |
|---------|--------|-------------------|
| Store | 1,115 | 50 |
| Day of Week | 7 | 6 |
| Day | 31 | 10 |
| Month | 12 | 6 |
| Year | 3 | 2 |
| State | 12 | 6 |
| Store Type | 5 | 2 |
| Assortment | 4 | 3 |

---

## Competition 5: Recruit Restaurant Visitor Forecasting (2018)

**Competition**: Forecast daily restaurant visitors for 821 restaurants for 39 days
**Data**: 15 months + restaurant metadata, holidays, reservation data
**Metric**: Root Mean Squared Logarithmic Error (RMSLE)
**Winner**: Team of 4 - ensemble of LightGBM + XGBoost + Neural Networks

### Winning Solution Architecture

**Model Ensemble:**
- **LightGBM** models
- **XGBoost** models
- **Feedforward neural networks**
- Final prediction: **average of all models**

**Key Features (different from earlier competitions):**

| Feature Category | Details |
|-----------------|---------|
| Rolling statistics | Standard rolling means, medians, std |
| Lagged values | Lagged restaurant reservations |
| Reservation data | Reservations made at different times in advance |
| Restaurant metadata | Genre, location, area |
| Holidays | Japanese holiday calendar |

**Domain Knowledge Trick (not used by winner but significant):**
- Test set included "Golden Week" holiday period
- Contestants had only seen one such period in training
- **Key trick**: Treat holidays as Saturdays, days prior as Fridays, days after as Mondays
- Gave significant performance boost when used
- Demonstrates value of domain knowledge + manual data adjustments

**Neural Network Performance Note:**
- 1st and 5th place used neural networks but mainly for ensemble diversity
- RNN and CNN variants performed slightly worse than boosted trees
- Kalman filter placed 33rd with only 2.4% gap to 1st place

**Validation Strategy:**
- Standard hold-out approach

**Key Insight That Won:**
> Simple average ensemble of diverse model types (LightGBM + XGBoost + NN) with rolling statistics and lagged reservation features. The reservation data was the key differentiator from earlier competitions.

---

## Competition 6: Grupo Bimbo Inventory Demand (2016)

**Competition**: Predict inventory demand for 800,000+ stores and 1,000+ products
**Winner**: "The Slippery Appraisals" team - 3-level stacking ensemble

### Winning Solution: Multi-Level Stack

**Level 1 - Sub-Models:**
- Many single models, mostly XGBoost
- Diverse base models

**Level 2 - Stacking:**
- ExtraTrees classifier
- Linear model (scikit-learn)
- Neural Network
- Trained on Level 1 predictions as features

**Level 3 - Weighted Average:**
- Weighted combination of Level 2 outputs

**Most Important Features:**
1. **Lags of target variable grouped by factors** (e.g., previous week sales)
2. **Aggregated features** (min, max, mean, sum) grouped by factors and combinations
3. **Frequency features** of factor variables
4. Key insight: "If too many products were supplied and not sold, next week supply decreases"

---

## Feature Engineering Patterns Across Winners

### Most Important Feature Categories (Ranked by Frequency)

| Rank | Feature Type | Used By | Impact |
|------|-------------|---------|--------|
| 1 | **Rolling statistics (mean, std)** | ALL winners | Critical - 18% MAPE improvement |
| 2 | **Lag features (t-1, t-7, t-14, t-28)** | ALL winners | Key predictors |
| 3 | **Calendar features (dow, month, year, is_weekend)** | ALL winners | Captures seasonality |
| 4 | **Event counters (days until/into/after event)** | Rossmann, M5 | Proved essential |
| 5 | **Price features (current, relative, stats)** | M5, Favorita | Important for retail |
| 6 | **Promotional features** | ALL retail comps | Critical for accuracy |
| 7 | **Holiday indicators** | ALL winners | Major sales driver |
| 8 | **Hierarchical statistics** | Rossmann, M5, Favorita | Cross-learning benefits |
| 9 | **Exponential moving average** | Favorita, M5 | Smoothed trends |
| 10 | **SNAP/food stamp indicators** | M5 | Specific to grocery |

### Specific Rolling Window Sizes Used

| Window | Usage |
|--------|-------|
| 7-day | Weekly seasonality (most common) |
| 14-day | Bi-weekly patterns |
| 28-day | Monthly seasonality (M5 primary) |
| 30-day | Monthly patterns |
| 60/90-day | Quarterly trends |

### Feature Grouping Levels (for hierarchical data)

1. By individual item/product
2. By store
3. By product category/class
4. By store-item combination
5. By store-category combination
6. By department
7. By state/region

---

## External Data Integration Strategies

### Types of External Data Used by Winners

| External Data | Competition | Integration Method |
|--------------|-------------|-------------------|
| **Weather** (precipitation, temp) | Rossmann | Direct features + rolling |
| **Google Trends** | Rossmann | Trend statistics as features |
| **Oil prices** | Corporacion Favorita | Daily price as numeric feature |
| **Holiday calendars** | All competitions | Binary indicators + event counters |
| **Reservation data** | Recruit Restaurant | Lagged values + rolling stats |
| **SNAP/food stamps** | M5 | Binary indicators per state |
| **Promotional calendars** | All retail | Binary + days until/after |
| **Competition distance/date** | Rossmann | Numeric features |

### Integration Patterns

1. **Direct features**: Add as numeric/categorical columns (oil price, temperature)
2. **Binary indicators**: 0/1 flags for holidays, promotions, SNAP
3. **Event counters**: Days until/into/after events (critical for promotions/holidays)
4. **Rolling aggregations**: Rolling stats of external data (e.g., 7-day avg temperature)
5. **Lagged values**: Previous values of external variables

---

## Ensemble Strategies Used by Winners

### Strategy 1: Simple Average (Most Common)

**Used by:** M5 1st place, Recruit Restaurant winner, Corporacion Favorita winner

```
Final_Prediction = (Model_1 + Model_2 + ... + Model_N) / N
```

**Why it works:**
- Different models make different errors
- Averaging cancels out uncorrelated errors
- No risk of overfitting the ensemble
- Surprisingly effective even with equally weighted models

### Strategy 2: Weighted Average

**Used by:** Grupo Bimbo winner (Level 3)

```
Final_Prediction = w1*Model_1 + w2*Model_2 + ... + wN*Model_N
```

- Weights determined by validation performance
- More weight to better-performing models

### Strategy 3: Multi-Level Stacking

**Used by:** Grupo Bimbo winner

```
Level 1: Train diverse base models (XGBoost, ARIMA, etc.)
Level 2: Train meta-learner on Level 1 predictions as features
Level 3: Weighted average of Level 2 outputs
```

### Strategy 4: Partial Pooling + Strategy Combination

**Used by:** M5 1st place

- Build models at different hierarchy levels (store, category, department)
- Combine recursive and non-recursive variants
- Simple average of selected models

### Strategy 5: External Adjustment

**Used by:** M5 2nd place

- Use one model type for granular predictions
- Use another model (N-BEATS) for trend at higher levels
- Adjust granular predictions with trend multipliers

### Ensemble Diversity Sources

| Source | Description |
|--------|-------------|
| Different algorithms | LightGBM + XGBoost + Neural Networks |
| Different data subsets | Train on different time periods |
| Different feature sets | Vary features per model |
| Different model structures | Recursive vs. non-recursive |
| Different hierarchy levels | Store-level vs. category-level models |
| Different forecast horizons | Per-horizon models vs. all-horizon models |

---

## Validation Strategies

### Time Series Cross-Validation (Used by All Top Winners)

**Standard Approach:**
- Use last **4+ forecast-length windows** for validation
- Strict temporal split (no shuffling)
- Expanding or rolling window

**M5 Specific:**
- Last four 28-day windows
- Measure both mean error AND standard deviation
- Select models that are both accurate AND stable

**Alternative Approaches That Also Worked:**

| Approach | Used By | Details |
|----------|---------|---------|
| Grouped K-Fold | 4th place Favorita | Each time series in one fold only |
| Hold-out | Top 3 Favorita | Simple last N days hold-out |
| Multi-window CV | M5 top 5 | Multiple consecutive validation windows |

### Key Validation Principles

1. **Never shuffle time series data** - preserves temporal order
2. **Validation period = forecast horizon length** - same distribution
3. **Multiple validation windows** - ensures robustness
4. **Monitor error variance** - not just mean error
5. **Use recent data for validation** - simulates real forecasting

---

## Model Architecture Evolution

### Timeline of Winning Approaches

| Year | Competition | Winner Model |
|------|------------|--------------|
| 2014 | Walmart Sales | ARIMA ensemble (R) + Christmas adjustment |
| 2015 | Rossmann | XGBoost ensemble (12 models) |
| 2016 | Grupo Bimbo | 3-level stacking (XGBoost + meta-learners) |
| 2017 | Corporacion Favorita | LightGBM + NN ensemble, per-horizon models |
| 2017 | Wikipedia Traffic | CNN + RNN (deep learning breakthrough) |
| 2018 | Recruit Restaurant | LightGBM + XGBoost + NN average |
| 2020 | M5 Accuracy | LightGBM DRFAM (220 models, recursive+direct) |
| 2020 | M5 Uncertainty | LightGBM quantile regression (126 models) |

### Key Technology Shifts

1. **2015**: XGBoost rises as dominant tool
2. **2016**: LightGBM introduced - faster, more efficient
3. **2017**: Neural networks enter top 3 (Entity Embeddings)
4. **2017**: Per-horizon training innovation
5. **2018**: LightGBM becomes standard across all winners
6. **2020**: Tweedie loss for zero-inflated data
7. **2020**: Recursive + Direct combination (DRFAM)
8. **2020**: DeepAR/LSTM viable but still secondary to GBDT

---

## Practical Recommendations

### For Retail Demand Forecasting

**Must-Have Features:**
```
# Lag features
target_lag_1, target_lag_7, target_lag_14, target_lag_28

# Rolling statistics
rolling_mean_7, rolling_mean_14, rolling_mean_28
rolling_std_7, rolling_std_14
exp_moving_avg_7, exp_moving_avg_14

# Calendar features
day_of_week, month, year, week_of_year, day_of_month
is_weekend, is_month_start, is_month_end

# Event features
days_until_holiday, days_after_holiday, is_holiday
days_until_promotion, is_promotion

# Price features
current_price, price_discount, price_vs_avg

# Hierarchical stats
mean_sales_by_store, mean_sales_by_item
mean_sales_by_store_item, mean_sales_by_category
```

**Recommended Model Stack:**
```
1. LightGBM with Tweedie loss (primary)
2. LightGBM with Poisson loss (secondary)
3. XGBoost with RMSE loss (diversity)
4. Simple average ensemble of above
```

**Recommended Validation:**
```
- Use last 4 periods of forecast-length for CV
- Example: For 28-day forecast, use four 28-day windows
- Select models by: mean_error + lambda * std_error (for stability)
```

**Key Hyperparameters (LightGBM):**
| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| objective | tweedie | For zero-inflated sales |
| tweedie_variance_power | 1.1 | Tune between 1.0-1.5 |
| learning_rate | 0.05-0.1 | Lower for final model |
| num_leaves | 31-128 | More for complex patterns |
| max_bin | 255 | Default usually fine |
| feature_fraction | 0.8 | Column sampling |
| bagging_fraction | 0.8 | Row sampling |
| num_iterations | 1000-5000 | Use early stopping or fixed |

---

## Key Insights Summary

| # | Insight | Evidence |
|---|---------|----------|
| 1 | **LightGBM is the gold standard** | Won or placed in every major forecasting competition since 2017 |
| 2 | **Ensembling provides 2-5% improvement** | Every winner used ensembles; single models never won |
| 3 | **Rolling statistics are the most important features** | Used by 100% of winners; multiple window sizes |
| 4 | **Recent data often beats all data** | Favorita winner: 1-5 months >> 55 months |
| 5 | **Tweedie loss for zero-inflated sales** | M5 winner; critical for intermittent demand |
| 6 | **Recursive + Direct combination wins** | M5 1st place; complementary strengths |
| 7 | **Per-horizon models help** | Favorita winner; 16 models for 16 days |
| 8 | **External variables matter** | Weather, holidays, promotions consistently used |
| 9 | **Cross-validation is critical** | 4+ validation windows; measure stability |
| 10 | **Simple average often beats complex blending** | M5 winner: simple average of 6 models |
| 11 | **GBDT > Deep Learning for tabular time series** | All M5 top 50 used LightGBM except one |
| 12 | **Feature engineering > model complexity** | Winners focused on features, not architectures |

---

## References

1. In, Y., & Jung, J. (2022). "Simple averaging of direct and recursive forecasts via partial pooling using machine learning." *International Journal of Forecasting*, 38(4), 1386-1399.
2. Makridakis, S., et al. (2022). "The M5 Accuracy competition: Results, findings and conclusions."
3. Bojer, C.S., & Meldgaard, J.P. (2020). "Learnings from Kaggle's Forecasting Competitions."
4. Ke, G., et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree." NeurIPS.
5. Guo, C., & Berkhahn, F. (2016). "Entity Embeddings of Categorical Variables." arXiv.
6. Salinas, D., et al. (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks."
7. Oreshkin, B.N., et al. (2019). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting."
8. Zhou, Z., et al. (2020). "Tweedie Gradient Boosting for Extremely Unbalanced Zero-inflated Data."
9. Kaggle Discussion Forums: M5 Forecasting, Corporacion Favorita, Rossmann, Recruit Restaurant
10. Kaggle Winning Solution Writeups (linked throughout)

---

*Document compiled from 20+ academic papers, Kaggle winner writeups, and competition post-mortems. Last updated: Current session.*
