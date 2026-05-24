# Dimension 7: Ensemble Strategies & Model Stacking for Forecasting

## Executive Summary

Ensembling is the single most important technique for achieving top-tier forecasting performance. Across Kaggle competitions, M-competitions, and academic benchmarks, ensemble methods consistently outperform individual models by 5-20%. For our problem with only 2 months of data, the recommended approach is a **carefully constrained weighted ensemble** (simple average or inverse-error-weighted) with 3-5 diverse base models, avoiding complex multi-layer stacking due to overfitting risk. The evidence overwhelmingly supports combining **GBDT + Statistical models** for short time series, with neural networks added only if sufficient diversity can be ensured without inflating model count.

---

## 1. Key Findings

### Finding 1: Multi-Layer Stacking Achieves State-of-the-Art Results

```
Claim: Multi-layer stacking consistently outperforms simple averaging and single-layer stacking across 50 real-world datasets, achieving the best Elo ratings and average ranks [^533^]
Source: Multi-layer Stack Ensembles for Time Series Forecasting (Bosch et al., AWS)
URL: https://arxiv.org/abs/2511.15350v1
Date: 2025-11-19
Excerpt: "Multi-layer stacking achieves the highest accuracy. This result is consistent with our earlier observations: since no single L2 model performs best across all datasets, combining them in an L3 ensemble leads to improved overall accuracy."
Confidence: HIGH
```

**Key results from the benchmark (50 datasets, 90K time series):**

| Method | Elo Rating | Champion Count | Avg Rank | Avg Relative Error |
|--------|-----------|----------------|----------|-------------------|
| Median (baseline) | 1000 | 3 | 5.92 | 1.000 |
| Model selection | 1049 | 9 | 5.43 | 1.001 |
| Performance-weighted avg | 1130 | 3 | 4.60 | 0.963 |
| Greedy ensemble selection | 1191 | 4 | 3.92 | 0.952 |
| Linear model (stacking) | 1220 | 3 | 3.62 | 0.947 |
| Nonlinear model | 1046 | 4 | 5.43 | 1.011 |
| **Multi-layer stacking** | **1306** | **20** | **2.81** | **0.945** |

**Critical caveat for our problem:** This benchmark used datasets with at least 8x forecast horizon observations. With only 2 months (~60 days) of data, aggressive stacking can overfit severely.

### Finding 2: The "Forecast Combination Puzzle" — Simple Average is Surprisingly Robust

```
Claim: Simple equal-weighted averaging often outperforms theoretically optimal weighting schemes, a phenomenon known as the "forecast combination puzzle" [^504^]
Source: Forecast combinations: an over 50-year review (Stock & Watson)
URL: https://arxiv.org/pdf/2205.04216v1
Date: 2022
Excerpt: "The simple average with equal weights often outperforms more complicated weighting schemes... empirically the simple average has been continuously found to dominate more complicated approaches to combining forecasts."
Confidence: HIGH
```

**Why simple averaging works:**
- Estimation error in "optimal" weights often exceeds the benefit of optimality
- Equal weights are maximally robust to structural breaks and regime shifts
- Simple averaging reduces variance without introducing weight estimation bias
- With limited data, the covariance matrix of forecast errors is poorly estimated, making "optimal" weights unreliable

**However**, the multi-layer stacking paper [^533^] notes this puzzle applies mainly to "small datasets, statistical models, and point forecasts" — with modern ML models and larger datasets, learned aggregation methods significantly outperform simple averages.

### Finding 3: M5 Competition Winners Used LightGBM Ensembles with Simple Averaging

```
Claim: The M5 Accuracy winner used an equal-weighted combination of 220 LightGBM models trained at different aggregation levels [^462^]
Source: The M5 Accuracy Competition: Results, Findings and Conclusions
URL: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
Date: 2021
Excerpt: "The winner considered an equal weighted combination (arithmetic mean) of various LightGBM models that were trained to produce forecasts for the product-store series using data per store (10 models), store-category (30 models), and store-department (70 models)."
Confidence: HIGH
```

**Key insight:** The M5 winner achieved ~20% improvement over statistical benchmarks using **equal weighting** of diverse LightGBM variants, not complex stacking. The diversity came from:
- Different aggregation levels (store, store-category, store-department)
- Recursive vs. non-recursive forecasting approaches
- Different learning approaches and train sets

### Finding 4: Kaggle Grandmaster Stacking Strategy — 500 Models to 75 in 3-Level Stack

```
Claim: Kaggle Grandmaster Chris Deotte won by exploring 500 models, selecting 75 diverse Level 1 models, and combining them in a 3-level stack [^522^]
Source: NVIDIA Blog — Grandmaster Pro Tip
URL: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/
Date: 2025-06-12
Excerpt: "The final solution uses 75 Level 1 models that were chosen from 500 experimental models. The secret to building a strong stack is to explore many Level 1 models."
Confidence: HIGH
```

**Architecture:** Level 1 (75 diverse models: GBDT, NN, SVR, KNN) → Level 2 (multiple stackers) → Level 3 (simple averaging of L2 outputs)

### Finding 5: For Short Time Series, Start with 3-5 Models Maximum

```
Claim: With fewer than 1,000 samples, simpler models or single algorithms often generalize better, and stacking complexity must be carefully controlled [^454^]
Source: Stacking Ensemble: Practical Guide
URL: https://mcpanalytics.ai/articles/stacking-ensemble-practical-guide-for-data-driven-decisions
Date: 2025-12-26
Excerpt: "Avoid stacking if you face: Limited Training Data: With fewer than 1,000 samples, simpler models or single algorithms often generalize better."
Confidence: HIGH
```

```
Claim: A rule of thumb for linear models is at least 10 samples per predictor variable [^537^]
Source: A review of machine learning with small and limited data
URL: https://link.springer.com/article/10.1186/s40537-025-01346-9
Date: 2026-01-16
Excerpt: "A rule of thumb for linear models is the 10:1 rule: at least 10 samples per predictor variable to avoid overfitting... regression tasks with n < 100 often struggle with high-dimensional features."
Confidence: MEDIUM
```

---

## 2. Major Techniques

### 2.1 Weighted Averaging

#### Simple Average (Equal Weights)
The most robust method when data is limited:

```
ŷ = (1/M) * Σ ŷ_m
```

**When to use:** Limited validation data, high uncertainty about model quality, or when models have similar expected performance.

```
Claim: FFORMA deviates significantly from simple averaging — the latter produces a 14% increase in error for the same pool of methods [^463^]
Source: FFORMA: Feature-based Forecast Model Averaging (M4 Competition 2nd place)
URL: https://robjhyndman.com/papers/fforma.pdf
Date: 2019
Excerpt: "FFORMA deviates significantly from simple averaging. The latter produces a 14% increase in error for the same pool of methods."
Confidence: HIGH (note: FFORMA was trained on thousands of time series, not 2 months)
```

#### Inverse-Error Weighted

```
w_m = (1/MAE_m) / Σ(1/MAE_j)
```

**Variants (implemented in AutoGluon) [^477^]:**
- `inv`: weights ∝ 1/S (inverse of validation score)
- `sqrt`: weights ∝ sqrt(1/S) — **default in AutoGluon, most robust**
- `sq`: weights ∝ (1/S)^2 — aggressive, can overfit

```
Claim: AutoGluon uses sqrt(1/S) as the default weighting scheme, balancing between equal weighting and aggressive performance weighting [^477^]
Source: AutoGluon TimeSeries Documentation
URL: https://auto.gluon.ai/dev/_modules/autogluon/timeseries/models/ensemble/weighted/basic.html
Date: 2025
Excerpt: "'sqrt' computes weights in proportion to sqrt(1/S). This is the default."
Confidence: HIGH
```

#### Bates-Granger Inverse Variance Weights

```
w_m = σ̂_m^(-2) / Σ σ̂_j^(-2)
```

Where σ̂_m^2 is the historical forecast error variance of model m. Bates and Granger (1969) showed these are optimal when forecast errors are uncorrelated.

```
Claim: Combinations ignoring correlations are more successful than those attempting to take account of correlations [^513^]
Source: Forecast combinations: an over 50-year review
URL: https://arxiv.org/pdf/2205.04216
Date: 2022
Excerpt: "Their extensive results demonstrated that combinations ignoring correlations are more successful than those attempting to take account of correlations."
Confidence: HIGH
```

### 2.2 Stacking (Super Learning)

Stacking trains a meta-learner on base model predictions to learn optimal combination weights.

#### Single-Layer Stacking
```
Base Models f_1, ..., f_M → Predictions ŷ_1, ..., ŷ_M → Stacker g(ŷ_1, ..., ŷ_M) → Final prediction
```

**Key implementation detail — out-of-fold predictions [^533^]:**

```
Claim: Stacker models should be trained on out-of-fold data using K-fold time series cross-validation [^533^]
Source: Multi-layer Stack Ensembles for Time Series Forecasting
URL: https://arxiv.org/abs/2511.15350v1
Date: 2025
Excerpt: "For each fold k, remove the last j windows of size H from each time series, train base models, make H-step-ahead predictions... collect the training set for the stacker model."
Confidence: HIGH
```

**Recommended meta-learners for time series (ranked by performance in [^388^]):**
1. **Random Forest** — most accurate meta-learner, little sensitivity to training set size
2. **k-Nearest Neighbors** — non-parametric, captures local patterns
3. **MLP (Neural Network)** — nonlinear combination capability
4. **LSTM** — captures temporal relationships in base predictions
5. **Linear Regression** — simplest, most interpretable

```
Claim: Random Forest as meta-learner generated the most accurate predictions and displayed little sensitivity to the size of the training set [^388^]
Source: Combining Forecasts using Meta-Learning
URL: https://arxiv.org/html/2504.08940v1
Date: 2025-04-11
Excerpt: "Among the meta-models, RF stood out by generating the most accurate predictions and displaying little sensitivity to the size of the training set."
Confidence: HIGH
```

#### Multi-Layer Stacking

```
L1: Base forecasters {f_m} → predictions {ŷ_m}
L2: Multiple stacker models {g_c} → combined predictions {ŷ_c}
L3: Aggregator s({ŷ_c}) → final prediction
```

The AWS paper [^533^] demonstrated that L3 (greedy ensemble selection over L2 outputs) consistently beats any single L2 stacker.

### 2.3 Blending

Blending uses a hold-out validation set to train the meta-learner.

```
Claim: Blending specifically refers to ensemble selection applied to a holdout validation set distinct from the training data [^407^]
Source: HAPEns: Hardware-Aware Post-Hoc Ensembling
URL: https://arxiv.org/html/2603.10582v1
Date: 2026-03-11
Excerpt: "The term blending also appears in this context, but it specifically refers to ensemble selection applied to a holdout validation set distinct from the training data."
Confidence: HIGH
```

**Blending vs. Stacking:**
- **Stacking:** Uses K-fold cross-validation to generate out-of-fold predictions for meta-learner training. More data-efficient.
- **Blending:** Uses a single held-out validation set. Simpler but requires enough data to spare a validation set.

**For our problem (2 months data):** Blending is risky because holding out data leaves very little for training. Prefer stacking with K-fold time series CV.

### 2.4 Greedy Ensemble Selection (Caruana's Algorithm)

```
Claim: GES iteratively adds models to the ensemble that lead to the largest improvement, with replacement [^404^]
Source: HAPEns: Hardware-Aware Post-Hoc Ensembling
URL: https://arxiv.org/html/2603.10582v1
Date: 2026-03-11
Excerpt: "ES as introduced by Caruana et al. (2004) is a forward selection algorithm that greedily constructs an ensemble by iteratively adding the model that improves the predictive performance of the ensemble the most."
Confidence: HIGH
```

**Algorithm:**
1. Initialize: empty ensemble (or best single model)
2. For each iteration:
   - Try adding each candidate model to current ensemble
   - Select the model that maximizes validation performance
   - Add it (with replacement — models can be added multiple times)
3. After ensemble_size iterations, normalize weights

```python
# From AutoGluon implementation [^479^]
class GreedyEnsemble:
    def fit(self, predictions_per_window, data_per_window):
        weights = {model: 0 for model in model_names}
        for iteration in range(ensemble_size):
            best_model = argmin_model(validation_loss_with(weights + candidate))
            weights[best_model] += 1
        # Normalize to get final weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
```

**Default ensemble_size = 100** in AutoGluon. For small datasets, reduce to 20-50.

---

## 3. Implementation Details

### 3.1 Recommended Ensemble Architecture for Our Problem

Given **only ~60 days of data**, the recommended architecture is:

```
Level 1 (3-5 diverse base models):
  ├── LightGBM (with different feature sets)
  ├── XGBoost or CatBoost
  ├── Statistical model (ARIMA/ETS/Prophet)
  └── Optional: Simple neural network (MLP)

Level 2 (Ensemble combination):
  └── Simple average OR Inverse-MAE-weighted average
  
NO Level 3 for our data size — too much complexity
```

### 3.2 Code Examples

#### Basic Weighted Ensemble

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

def simple_average(predictions_dict):
    """Equal-weighted ensemble. Most robust with limited data."""
    preds = np.array(list(predictions_dict.values()))
    return np.mean(preds, axis=0)

def inverse_mae_weighted(predictions_dict, y_true):
    """Weight by inverse validation MAE."""
    weights = {}
    for name, pred in predictions_dict.items():
        mae = mean_absolute_error(y_true, pred)
        weights[name] = 1.0 / (mae + 1e-8)
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    # Compute weighted prediction
    result = np.zeros_like(list(predictions_dict.values())[0])
    for name, pred in predictions_dict.items():
        result += weights[name] * pred
    return result, weights

def sqrt_inverse_score_weighted(predictions_dict, scores):
    """
    AutoGluon-style: weight by sqrt(1/score).
    scores: dict of {model_name: validation_score} 
            (lower is better, e.g., MAE)
    """
    weights = {}
    for name, score in scores.items():
        weights[name] = np.sqrt(1.0 / (score + 1e-8))
    
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}
```

#### Time Series Stacking with Out-of-Fold Predictions

```python
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

def generate_oof_predictions(model, X, y, n_splits=3):
    """
    Generate out-of-fold predictions for stacking.
    Uses walk-forward validation (time series aware).
    """
    n = len(y)
    oof_preds = np.zeros(n)
    
    # For very short series, use fewer splits
    fold_size = n // (n_splits + 1)
    
    for i in range(n_splits):
        # Train on first (i+1) folds, predict on next
        train_end = (i + 1) * fold_size
        test_end = min((i + 2) * fold_size, n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_test = X[train_end:test_end]
        
        model.fit(X_train, y_train)
        oof_preds[train_end:test_end] = model.predict(X_test)
    
    return oof_preds

def stack_models(base_predictions_dict, y_true, meta_learner='ridge'):
    """
    Train a meta-learner on base model predictions.
    
    Parameters:
    -----------
    base_predictions_dict: {model_name: oof_predictions_array}
    y_true: true target values
    meta_learner: 'ridge', 'rf', or 'linear'
    """
    X_meta = np.column_stack(list(base_predictions_dict.values()))
    
    if meta_learner == 'ridge':
        # RidgeCV automatically selects alpha via CV
        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    elif meta_learner == 'rf':
        meta = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    else:
        from sklearn.linear_model import LinearRegression
        meta = LinearRegression()
    
    meta.fit(X_meta, y_true)
    
    # Return fitted meta-learner and feature importances (weights)
    return meta, dict(zip(base_predictions_dict.keys(), 
                          meta.coef_ if hasattr(meta, 'coef_') else 
                          meta.feature_importances_))
```

#### Greedy Ensemble Selection (Caruana)

```python
def greedy_ensemble_selection(predictions_dict, y_val, ensemble_size=25):
    """
    Caruana's greedy forward selection algorithm.
    For small datasets, use ensemble_size=20-50.
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Initialize with best single model
    best_mae = float('inf')
    for name, pred in predictions_dict.items():
        mae = mean_absolute_error(y_val, pred)
        if mae < best_mae:
            best_mae = mae
            ensemble = {name: 1}
    
    # Greedily add models
    for _ in range(ensemble_size - 1):
        best_mae = float('inf')
        best_addition = None
        
        for name in model_names:
            # Try adding this model
            trial = ensemble.copy()
            trial[name] = trial.get(name, 0) + 1
            
            # Compute weighted prediction
            total = sum(trial.values())
            pred = np.zeros_like(y_val, dtype=float)
            for m, w in trial.items():
                pred += (w / total) * predictions_dict[m]
            
            mae = mean_absolute_error(y_val, pred)
            if mae < best_mae:
                best_mae = mae
                best_addition = name
        
        if best_addition:
            ensemble[best_addition] = ensemble.get(best_addition, 0) + 1
    
    # Normalize to weights
    total = sum(ensemble.values())
    return {k: v/total for k, v in ensemble.items()}
```

### 3.3 Parameters and Settings

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| Number of base models | 3-5 | More models risk overfitting with limited data |
| Ensemble size (GES) | 20-50 | Reduce from default 100 for small datasets |
| Meta-learner | RidgeCV or RF (max_depth=3) | Regularized to prevent overfitting |
| K-fold for OOF | 3 folds | With ~60 days, 3 folds gives ~20 days per validation |
| Weight scheme | sqrt inverse or simple average | Most robust for small samples |

---

## 4. What Works

### ✅ Simple Average as Baseline

The simple average is the most robust ensemble method when data is limited. It:
- Avoids weight estimation error entirely
- Reduces variance without adding parameters
- Is maximally robust to structural changes

```
Claim: Simple averaging has been shown to often outperform individual forecasting models [^533^]
Source: Multi-layer Stack Ensembles for Time Series Forecasting
URL: https://arxiv.org/abs/2511.15350v1
Date: 2025
Excerpt: "Simple averaging has been shown to often outperform individual forecasting models"
Confidence: HIGH
```

### ✅ GBDT + Statistical Model Combination

The most consistently effective combination for demand forecasting:

```
Claim: The winning M5 solution used LightGBM ensembles, and GBDT+NN combinations are standard in Kaggle forecasting competitions [^462^]
Source: M5 Accuracy Competition Results
URL: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
Date: 2021
Excerpt: "Most of the methods examined utilized LightGBM... The use of ensembling of multiple XGBoost models provided a performance boost of around 5% over the best single model."
Confidence: HIGH
```

### ✅ Ridge Regression as Meta-Learner

Ridge regression (L2 regularized) is the safest meta-learner choice:

```
Claim: Ridge regression as meta-learner stabilizes predictions compared to standalone models [^543^]
Source: Baseball prediction Kaggle solution
URL: https://github.com/vivian-sketch58/baseball-prediction
Date: 2026
Excerpt: "The Ridge meta-learner stabilizes predictions compared to standalone models, which are more sensitive to historical outliers."
Confidence: MEDIUM
```

### ✅ Out-of-Fold Cross-Validation

Always generate meta-learner training data via cross-validation:

```
Claim: K-fold cross-validation separates training and validation data, ensuring base models predict unseen samples for unbiased estimation [^415^]
Source: Photovoltaic Power Prediction Framework
URL: https://www.mdpi.com/1996-1073/18/17/4644
Date: 2025
Excerpt: "If models are trained directly on the complete training set and predict the same data, their predictions become highly correlated with true labels... K-fold cross-validation separates training and validation data."
Confidence: HIGH
```

### ✅ Diversity Through Different Feature Sets

The M5 winner introduced diversity by training the same algorithm (LightGBM) on different feature subsets and aggregation levels. This is more data-efficient than using fundamentally different algorithms.

### ✅ Per-Item Ensemble Weights (AutoGluon)

For heterogeneous time series with different patterns:

```
Claim: Per-item greedy ensemble fits separate weighted ensembles for each individual time series [^479^]
Source: AutoGluon Documentation
URL: https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-ensembles.html
Date: 2025
Excerpt: "This ensemble applies the greedy Ensemble Selection algorithm independently to each time series in the dataset, allowing for customized model combinations."
Confidence: HIGH
```

---

## 5. What Doesn't Work

### ❌ Complex Multi-Layer Stacking with < 1000 Samples

With only 2 months of data, 3-level stacking is extremely likely to overfit:

```
Claim: With fewer than 1,000 samples, simpler models or single algorithms often generalize better [^454^]
Source: Stacking Ensemble Practical Guide
URL: https://mcpanalytics.ai/articles/stacking-ensemble-practical-guide-for-data-driven-decisions
Date: 2025
Excerpt: "Avoid stacking if you face: Limited Training Data: With fewer than 1,000 samples, simpler models or single algorithms often generalize better."
Confidence: HIGH
```

### ❌ Unregularized OLS as Meta-Learner

```
Claim: Classical OLS combinations may overfit and encounter degrees-of-freedom issues when the number of experts is large relative to the length of the forecast history [^508^]
Source: Regularized Ensemble Forecasting
URL: https://arxiv.org/html/2602.11379v1
Date: 2026
Excerpt: "Classical OLS combinations may overfit and encounter degrees-of-freedom issues when the number of experts is large relative to the length of the forecast history."
Confidence: HIGH
```

### ❌ Aggressive Performance Weighting (Inverse Squared)

The `sq` scheme in AutoGluon (weights ∝ 1/S²) can overfit by giving too much weight to the best validation performer.

### ❌ Too Many Base Models (>10 with limited data)

```
Claim: More models in an ensemble can sometimes increase overfitting, especially if additional models are highly correlated [^535^]
Source: Overfitting In Ensemble Methods
URL: https://www.meegle.com/en_us/topics/overfitting/overfitting-in-ensemble-methods
Date: 2026
Excerpt: "Adding more models to an ensemble can sometimes increase overfitting, especially if the additional models are highly correlated or overly complex."
Confidence: MEDIUM
```

### ❌ Using Future Information (Data Leakage in Stacking)

```
Claim: 10-fold cross-validation exhibits RMSE Gain values up to 20.5% due to data leakage in time series [^406^]
Source: Hidden Leaks in Time Series Forecasting
URL: https://arxiv.org/html/2512.06932v1
Date: 2025
Excerpt: "10-fold cross-validation exhibits RMSE Gain values up to 20.5% at extended lag steps. In contrast, 2-way and 3-way splits demonstrate greater robustness."
Confidence: HIGH
```

### ❌ Deep Neural Networks as Meta-Learners with Short Series

NN meta-learners require more data than tree-based or linear meta-learners. The study [^388^] found LSTM meta-learners were less robust than RF to training set size.

---

## 6. Competition Applications

### M5 Competition (2020) — Walmart Sales
- **Winner:** Equal-weighted ensemble of 220 LightGBM models
- **Diversity source:** Different aggregation levels + recursive vs. non-recursive
- **Key insight:** LightGBM ensembling beat all statistical benchmarks by 20%+

### Corporacion Favorita (2018) — Grocery Sales
- **Winner:** LightGBM + Neural Network + CNN ensemble
- **Key insight:** Training one model per forecast horizon improved accuracy
- **Data:** 55 months (winner used only 1-5 months of recent data)

```
Claim: The winner only used very recent data, electing to drop older observations based on validation dataset performance [^4^]
Source: Learnings from Kaggle's Forecasting Competitions (Aalborg University)
URL: https://vbn.aau.dk/ws/files/483966133/Kaggle_3_.pdf
Date: 2021
Excerpt: "The winner only used very recent data in the models, electing to drop older observations based on validation dataset performance. Thus, the final models used less than a full season of data."
Confidence: HIGH
```

### Kaggle Playground April 2025 — Podcast Listening Time
- **Winner (Chris Deotte):** 3-level stack with 75 L1 models chosen from 500
- **Architecture:** GBDT + NN + SVR + KNN → L2 stackers → L3 average
- **Key insight:** The secret is exploring MANY diverse models quickly

### AutoGluon-TimeSeries (2023)
- Combines 24 diverse base models using multi-layer stacking + weighted averaging
- Uses GreedyEnsemble (Caruana-style) as default L2/L3 combiner
- Supports per-item ensemble weights for heterogeneous datasets

---

## 7. Recommended Approach for Our Problem

### Given: Only 2 months (~60 days) of demand data

### Phase 1: Simple Ensemble (Start Here)

```python
# RECOMMENDED: 3-4 model simple/inverse-MAE-weighted ensemble
models = {
    'lightgbm_lags': lgbm_with_lag_features,      # GBDT with time features
    'lightgbm_agg': lgbm_with_aggregated_features, # Different feature set
    'prophet': prophet_model,                      # Statistical baseline
    'xgboost': xgboost_model,                      # Alternative GBDT
}

# Option A: Simple average (most robust)
final_pred = np.mean([pred for pred in models.values()], axis=0)

# Option B: Inverse-MAE weighted (if you have validation data)
weights = {}
for name, model in models.items():
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    weights[name] = 1.0 / mae

# Normalize and combine
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}
```

### Phase 2: Stacking (If More Data Becomes Available)

```python
# Use ONLY if you have at least 3-4 weeks for validation
# 1. Generate OOF predictions with 3-fold walk-forward CV
oof_preds = {}
for name, model in models.items():
    oof_preds[name] = generate_oof_predictions(model, X, y, n_splits=3)

# 2. Train Ridge meta-learner (regularized!)
meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
X_meta = np.column_stack(list(oof_preds.values()))
meta_learner.fit(X_meta, y)

# 3. For prediction: get base predictions, then meta-prediction
base_preds = np.column_stack([m.predict(X_test) for m in models.values()])
final_pred = meta_learner.predict(base_preds)
```

### Phase 3: Multi-Layer (Only with >6 months data)

Add only when data supports it:
- Multiple L2 stackers (Ridge, RF, linear)
- L3 greedy ensemble selection over L2 outputs

### Key Rules for Our Data Constraint:

1. **Maximum 5 base models** — each additional model adds overfitting risk
2. **No neural network meta-learners** — use RidgeCV or shallow RF
3. **Use 3-fold time series CV maximum** — preserve temporal order
4. **Prefer simple average** — inverse-MAE weights only if validation is reliable
5. **Diversity through feature subsets, not algorithms** — train same algorithm on different features
6. **Monitor ensemble vs. best single model** — if ensemble doesn't beat best single model by >2%, use the single model

### Expected Improvement

Based on competition evidence:
- Ensemble of 3-4 diverse models: **3-8% improvement** over best single model
- With only 2 months: expect **3-5%** (conservative)
- With simple averaging: more robust, less variance
- With inverse-MAE weighting: potentially better if validation is reliable

---

## 8. Sources

| # | Source | URL | Date |
|---|--------|-----|------|
| [^385^] | NVIDIA: Grandmaster Pro Tip — Winning with Stacking | https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ | 2025-05-22 |
| [^4^] | Learnings from Kaggle's Forecasting Competitions | https://vbn.aau.dk/ws/files/483966133/Kaggle_3_.pdf | 2021 |
| [^5^] | Learnings from Kaggle's Forecasting Competitions (arXiv) | https://arxiv.org/pdf/2009.07701 | 2020 |
| [^388^] | Combining Forecasts using Meta-Learning | https://arxiv.org/html/2504.08940v1 | 2025-04-11 |
| [^386^] | Global and Diverse Ensemble model for regression | https://www.sciencedirect.com/science/article/pii/S0925231225011920 | 2025-07-15 |
| [^391^] | AI for load forecasting: Stacking with diversity regularization | https://www.sciencedirect.com/science/article/abs/pii/S036054422202179X | 2024-09-29 |
| [^394^] | Super Learning from short time series | https://dl.acm.org/doi/abs/10.1007/s00180-024-01549-3 | 2024-09-23 |
| [^406^] | Hidden Leaks in Time Series Forecasting | https://arxiv.org/html/2512.06932v1 | 2025-12-07 |
| [^404^] | HAPEns: Hardware-Aware Post-Hoc Ensembling | https://arxiv.org/html/2603.10582v1 | 2026-03-11 |
| [^415^] | Photovoltaic Power Prediction: Multi-Stage Ensemble | https://www.mdpi.com/1996-1073/18/17/4644 | 2025-09-01 |
| [^419^] | RF-GRU-CNN-XGBoost Stacking: Data Leakage Prevention | https://www.mdpi.com/1911-8074/18/7/346 | 2025-06-21 |
| [^462^] | M5 Accuracy Competition Results | https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf | 2021 |
| [^463^] | FFORMA: Feature-based Forecast Model Averaging | https://robjhyndman.com/papers/fforma.pdf | 2019 |
| [^454^] | Stacking Ensemble: Practical Guide | https://mcpanalytics.ai/articles/stacking-ensemble-practical-guide-for-data-driven-decisions | 2025-12-26 |
| [^475^] | AutoGluon Weighted Ensemble | https://github.com/autogluon/autogluon/issues/3819 | 2023-12-14 |
| [^477^] | AutoGluon TimeSeries Ensemble Implementation | https://auto.gluon.ai/dev/_modules/autogluon/timeseries/models/ensemble/weighted/basic.html | 2025 |
| [^479^] | AutoGluon Forecasting Ensembles Documentation | https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-ensembles.html | 2025 |
| [^469^] | UK Electricity Demand Forecasting: Weighted Ensemble | https://github.com/aaliusama/uk-electricity-demand-forecasting | 2026-03-21 |
| [^471^] | TCN-LSTM-LightGBM Ensemble for Load Forecasting | https://www.sciencedirect.com/science/article/abs/pii/S0360544225003998 | 2025-04-24 |
| [^473^] | PSO-Enhanced Ensemble: LightGBM+XGBoost+DNN | https://www.preprints.org/manuscript/202501.1604 | 2025-01-22 |
| [^482^] | E-commerce Demand Prediction: Stacked Ensemble | https://www.atlantis-press.com/article/126016980.pdf | 2023 |
| [^503^] | Forecast combinations: an over 50-year review | https://ar5iv.labs.arxiv.org/html/2205.04216 | 2022 |
| [^504^] | Forecast combinations: 50-year review (PDF) | https://arxiv.org/pdf/2205.04216v1 | 2022 |
| [^506^] | γ-Relaxation: Forecast Combination and Portfolio Analysis | https://arxiv.org/pdf/2010.09477 | 2020 |
| [^508^] | Regularized Ensemble Forecasting | https://arxiv.org/html/2602.11379v1 | 2026 |
| [^510^] | Theoretical comparison of weight constraints in forecast combination | https://arxiv.org/html/2510.26456v1 | 2025 |
| [^533^] | Multi-layer Stack Ensembles for Time Series Forecasting | https://arxiv.org/abs/2511.15350v1 | 2025-11-19 |
| [^535^] | Overfitting In Ensemble Methods | https://www.meegle.com/en_us/topics/overfitting/overfitting-in-ensemble-methods | 2026-02-08 |
| [^537^] | A review of machine learning with small and limited data | https://link.springer.com/article/10.1186/s40537-025-01346-9 | 2026-01-16 |
| [^543^] | Baseball Prediction: Ridge Meta-Learner | https://github.com/vivian-sketch58/baseball-prediction | 2026-04-06 |
| [^459^] | State of Machine Learning Competitions 2024 | https://mlcontests.com/state-of-machine-learning-competitions-2024/ | 2025-02-25 |
| [^522^] | NVIDIA Blog: Chris Deotte Stacking | https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ | 2025-06-12 |
| [^143^] | Survey of ML Methods for Time Series Prediction (M5 section) | https://www.mdpi.com/2076-3417/15/11/5957 | 2025-05-26 |
| [^543^] | Baseball prediction: Ridge meta-learner | https://github.com/vivian-sketch58/baseball-prediction | 2026-04-06 |
