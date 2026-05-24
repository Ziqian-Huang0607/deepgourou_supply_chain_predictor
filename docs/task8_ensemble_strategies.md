# Task 8: Ensemble Strategies & Model Stacking for Demand Forecasting

## Executive Summary

Ensemble methods are the **dominant strategy** in demand forecasting competitions. In the M4 competition, ensemble forecasting models occupied the **first 12 places out of 17**. The M5 competition winner used a **simple equal-weighted combination of 220 LightGBM models**. For datasets with **<100 days of data**, research shows that **3-7 diverse models** provide optimal accuracy with diminishing returns beyond that. The key insight: **ensemble diversity matters more than individual model accuracy**, and **simple equal weighting often outperforms complex weight estimation** due to the "forecast combination puzzle."

---

## 1. What Wins: Ensemble Strategy in Demand Forecasting Competitions

### 1.1 M4 Competition (2018) - Key Findings

The M4 competition (100,000 time series, 61 methods) provided definitive evidence for ensemble dominance:

| Rank | Method | Approach | OWA Score |
|------|--------|----------|-----------|
| 1 | Smyl (Winner) | **Hybrid ETS + RNN** | 0.821 |
| 2 | Ata-best | **Hybrid ARIMA + Ata** (ensemble) | 0.837 |
| 3 | ProLogistica | Weighted ensemble of statistical methods | ~0.84 |
| Runner-up | FFORMA | **XGBoost meta-learner** over 9 base models | ~0.838 |

**Key finding**: "Ensemble forecasting models were the big winners, occupying the **first 12 places out of 17** in the competition" (Springer article on hybrid ensemble learning).

### 1.2 M5 Competition (2020) - Key Findings

The M5 Accuracy competition (Walmart retail data) marked a turning point:

| Rank | Method | Approach |
|------|--------|----------|
| 1 | YJ_STU (Winner) | **Equal-weighted ensemble of 220 LightGBM models** |
| 2 | 2nd place | LightGBM ensemble adjusted by N-BEATS multipliers |
| 3 | 3rd place | **Equal-weighted ensemble of 43 deep learning NNs** |
| 4 | 4th place | Non-recursive LightGBM per store |
| 5 | 5th place | Recursive LightGBM per department |

**Critical finding**: "LightGBM was used in **four of the top five models** in the M5 time series forecasting competition. It was the first M-competition where **pure ML methods achieved superior results** compared to simple statistical and more sophisticated methods." (Makridakis et al., IJF 2022)

**The winning recipe**: 220 LightGBM models trained on different aggregation levels (store: 10 models, store-category: 30 models, store-department: 70 models) x 2 variations (recursive + non-recursive) = **equal weighted average of 6 models per series**.

### 1.3 Kaggle Forecasting Competitions - Winner Patterns

A comprehensive review of 6 Kaggle forecasting competitions revealed consistent patterns:

| Competition | Winner | Key Ensemble Strategy | Improvement from Ensembling |
|-------------|--------|----------------------|---------------------------|
| Rossmann | XGBoost ensemble | Ridge trend + XGBoost + feature subsets | **~5%** over best single model |
| Wikipedia Traffic | RNN ensemble | Global RNN + SWA + multiple seeds | Multiple ensembling strategies |
| Corporacion Favorita | LightGBM + NN ensemble | Per-horizon models + 4-model ensemble | Significant boost from diversity |
| Restaurant Visitors | LightGBM + XGBoost + NN | Average of 3 different model families | Top method combination |

**From "Learnings from Kaggle's Forecasting Competitions"**: "Ensembles won all of the competitions... Global models were used by all competition winners... The use of ensembling of multiple XGBoost models provided a **performance boost of around 5%** over the best single model."

---

## 2. Model Stacking & Super Learning for Time Series

### 2.1 The Super Learner Algorithm

The Super Learner (stacking) approach consists of three phases:

1. **Setup**: Specify L base learners + a meta-learning algorithm (typically regularized regression)
2. **Train**: Train each base learner, perform k-fold CV to collect out-of-sample predictions, form a level-one dataset
3. **Predict**: Generate base learner predictions, feed into meta-learner for final output

**Key insight from research**: "Stacking never does worse than selecting the single best base learner on the training data. The biggest gains are usually produced when stacking base learners that have **high variability and uncorrelated predicted values**." (Hands-On ML with R)

### 2.2 FFORMA: Feature-Based Forecast Model Averaging

FFORMA (M4 runner-up) represents the state-of-the-art in meta-learning for forecast combination:

**How it works**:
1. Extract **42 time series features** (length, trend strength, seasonality strength, autocorrelations, etc.)
2. Train **XGBoost meta-learner** to predict combination weights from these features
3. Combine base model forecasts using learned weights

**Base model pool**: Naive, ETS, ARIMA, Theta, STL, Random Walk with Drift, etc.

**Results**:
- FFORMA achieved **10% lower error** than model selection approach
- Simple averaging produced **14% higher error** than FFORMA for the same pool
- Roughly **40% of series received equal-like weights**, 60% had dominant models
- Maximum increase in error when removing any single method: only **1%**

### 2.3 When Stacking Works Best

| Condition | Effectiveness |
|-----------|--------------|
| Base models have **diverse architectures** | High gains |
| Errors are **uncorrelated** | High gains |
| Base models have **similar individual accuracy** | Stacking > Selection |
| Small dataset (< 100 days) | Risk of meta-learner overfitting - use simple averaging |

---

## 3. Weighted Average Ensembles: The Forecast Combination Puzzle

### 3.1 The Puzzle: Why Simple Averaging Often Wins

A 50+ year body of research reveals a surprising finding:

> "Simple combination schemes - the mean, trimmed mean and median - work well... This fact has been called the **'forecast combination puzzle'**, since theory suggests it should be possible to improve upon simple combination forecasts." (Forecast combinations: an over 50-year review, 2022)

**Why simple averaging dominates**:
1. **Estimation error**: Optimal weights require estimation; simple averages have zero estimation error
2. **Parameter instability**: Optimal weights change over time; regression estimates "average" weights
3. **Bias-variance tradeoff**: Underfitting from equal weights vs. variance from weight estimation
4. **Empirical regularity**: When forecast errors have similar variances and positive correlations, gains from optimality are small

### 3.2 Inverse Error Weighting

Despite the puzzle, inverse error weighting remains widely used and effective:

```
Weight_i = (1 / MAE_i) / sum(1 / MAE_j) for all j
```

**Evidence**: "MASE-weighted combination assigns weights proportional to 1/MASE: better models get higher weights. A model with MASE=0.5 gets twice the weight of a model with MASE=1.0." Empirical results show **11% improvement** of weighted over simple averaging on financial forecasts.

**Implementation best practice**: Compute weights on a **validation set** (not training set) to avoid overfitting. Use exponential weighting if recent performance matters more.

### 3.3 Practical Weighting Strategies (Ranked by Complexity)

| Method | When to Use | Expected Gain |
|--------|------------|---------------|
| **Simple average** | < 100 days data, diverse models, similar accuracy | Baseline |
| **Trimmed mean** (remove extremes) | When some models occasionally fail | Small improvement |
| **Inverse MAE weight** | Validation data available, models differ in accuracy | 5-15% improvement |
| **Inverse variance weight** | Can estimate reliable error variances | Similar to inverse MAE |
| **Optimal regression weights** | Large sample, stable relationships | Often WORSE than simple |
| **Meta-learner (FFORMA)** | Large reference set available | Best when feasible |

---

## 4. LightGBM & XGBoost: The GBDT Ensemble Foundation

### 4.1 Why GBDT Dominates

Gradient Boosted Decision Trees have become the workhorse of forecasting competitions:

| Competition | GBDT Usage | Key Insight |
|-------------|-----------|-------------|
| M5 | 45 of top 50 used LightGBM | Superior without extensive preprocessing |
| Rossmann | XGBoost won | Feature engineering + ensemble = victory |
| Corporacion Favorita | LightGBM dominated | Faster iteration, easier experimentation |

**Why LightGBM specifically**:
- Fast training enabling more experimentation
- Good performance without extensive tuning
- Easy cross-learning (learning across different time series)
- Handles categorical features natively
- Tweedie loss effective for intermittent demand

### 4.2 GBDT Ensembling Strategies

**M5 Winner's approach**:
- 220 LightGBM models across 3 aggregation levels
- 2 variations per model (recursive + non-recursive)
- Each series forecast = average of **6 models**
- Used Tweedie distribution loss
- Cross-validated on last four 28-day windows

**Rossmann Winner's approach**:
- Multiple XGBoost models with variation through:
  - Different data subsets
  - Direct AND iterated predictions
  - Different feature subsets
- **~5% boost** from ensembling over best single model

### 4.3 Creating Diversity Within GBDT Ensembles

| Diversity Source | How | Effect |
|-----------------|-----|--------|
| Different data aggregation levels | Store / category / department | Different pattern exposure |
| Recursive vs non-recursive | One-step vs multi-step targets | Different error patterns |
| Different feature subsets | Vary rolling window lengths | Different temporal focus |
| Different seeds/hyperparameters | Random initialization | Variance reduction |
| Direct per-horizon models | Separate model for each forecast step | Horizon-specific learning |

---

## 5. Neural Network + GBDT Hybrid Approaches

### 5.1 The Hybrid Advantage

Combining neural networks with GBDT provides complementary strengths:

| Model Type | Strengths | Weaknesses |
|-----------|-----------|------------|
| **GBDT (LightGBM/XGBoost)** | Excellent on tabular features, fast, robust | Cannot extrapolate trends, poor at autoregressive patterns |
| **Neural Networks (RNN/Transformer)** | Good at sequential patterns, can extrapolate | Needs more data, harder to tune, less robust on tabular |

**Evidence from competitions**: "The winner used a relatively complex ensemble of models consisting of both gradient boosting models and neural network models" (Corporacion Favorita). "The combination of deep learning feature extraction with GBDT classification leverages the representational power of neural networks and the robustness of tree-based methods."

### 5.2 Winning Hybrid Architectures

**Architecture 1: GBDT + Residual NN (M4 Winner)**
- ETS (exponential smoothing) for trend/seasonality
- RNN (LSTM) for nonlinear residual patterns

**Architecture 2: Separate Per-Horizon Models**
- Train one LightGBM per forecast horizon
- Train one feedforward NN per forecast horizon
- Ensemble with another LightGBM (all horizons) + CNN

**Architecture 3: NN-STACK (Meta-learner)**
- Base models: ES-RNN, N-BEATS, statistical models
- Meta-learner: Neural network regression on base predictions
- Performs regression over single points of base learner forecasts

---

## 6. Why Ridge Regression Complements GBDT

### 6.1 The Core Problem: GBDT Cannot Extrapolate Trends

This is one of the most critical insights for demand forecasting:

> "Tree-based models **cannot extrapolate** beyond training data ranges. If test data contains feature values outside training bounds, predictions default to the nearest training leaf." (Practical Guide to Gradient Boosting)

For time series, this means:
- GBDT cannot predict a value higher than the maximum seen in training
- Linear trends (upward/downward) are inherently extrapolation problems
- The model essentially "flattens out" at the boundary

### 6.2 The Solution: Ridge + GBDT Decomposition

**The standard hybrid approach** used by competition winners:

```
Final Prediction = Linear_Trend (Ridge) + Nonlinear_Patterns (GBDT)
```

**Rossmann winner's implementation**:
1. **Component A (Trend)**: Ridge Linear Regression trained on deterministic `time_step` feature - captures global slope
2. **Component B (Seasonality & Residuals)**: LightGBM trained on residuals (Actual - Linear Trend) - captures complex patterns
3. **Final**: Sum of both components

**Why Ridge specifically**:
- **L2 regularization** prevents overfitting with limited data
- Fast to train, stable numerically
- Provides unbiased trend estimate
- Complements GBDT's nonlinear strength

**Alternative approach**: LightGBM's `linear_tree` option enables linear base learners within trees, allowing direct extrapolation. This was added specifically to address the trend extrapolation limitation.

### 6.3 Model Complementarity in Ensembles

| Model Family | Captures | Use For |
|-------------|----------|---------|
| **Ridge / Linear** | Global trends, smooth patterns | Trend extrapolation |
| **GBDT (LightGBM/XGBoost)** | Nonlinear interactions, local patterns | Seasonality, promotions, complex features |
| **Statistical (ETS/ARIMA)** | Autoregressive structure, seasonality | Baseline, small data |
| **Neural Network** | Long-range dependencies, embeddings | Sequential patterns, categorical embeddings |

---

## 7. Ensemble Diversity: How to Create Diverse Base Models

### 7.1 Why Diversity Matters

> "The more similar the predicted values are between the base learners, the less advantage there is to combining them." (Super Learner framework)

**Practical rule of thumb**: If two models have **error correlation > 0.7**, one is likely redundant.

### 7.2 Sources of Diversity for Demand Forecasting

| Diversity Dimension | Techniques | Impact |
|-------------------|------------|--------|
| **Algorithmic** | Linear + Tree + Neural Network + Statistical | Highest impact |
| **Feature engineering** | Different lag structures, rolling windows | Medium impact |
| **Data sampling** | Different training windows, bootstrap samples | Medium impact |
| **Target formulation** | Recursive vs non-recursive, different horizons | High impact |
| **Loss function** | MSE vs MAE vs Tweedie vs Quantile | Medium impact |
| **Data aggregation** | Store-level vs category-level vs global | High impact |

### 7.3 The Diversity-Based FFORMA Extension

Recent research proposes "Forecast with Forecasts: Diversity Matters":
- Instead of using time series features, use **forecast diversity** to determine weights
- Extract diversity matrix from out-of-sample forecasts
- Train combination model on diversity features
- Addresses FFORMA's limitation of requiring sufficient historical data

---

## 8. How Much Do Ensembles Help? Diminishing Returns Analysis

### 8.1 Empirical Evidence on Ensemble Size

Research on "The cost of ensembling" provides definitive guidance:

> "Accuracy improvements show **diminishing returns beyond three or four models**. Combining just two models is often sufficient to achieve optimal performance for accuracy-optimized ensembles." (2025 study on M5 and VN1 datasets)

**Key findings from the study**:
- ENSACC (accuracy-optimized): Diminishing returns after 3-4 models
- ENSTIME (efficiency-optimized): Continued improvements up to 5 models
- Small ensembles of 2-3 models achieve **near-optimal accuracy**
- Probabilistic forecasting benefits more from larger ensembles than point forecasting

### 8.2 Historical Pattern: Makridakis & Winkler (1983)

The original research found:
- Simple-average combinations become more accurate as component forecasts increase
- Gains follow a **sharply diminishing marginal returns** pattern
- The curve is approximately **hyperbolic** - most gains from first few models

### 8.3 Kaggle-Specific Estimates

| Source | Ensemble Improvement | Context |
|--------|---------------------|---------|
| Rossmann competition | **~5%** over best single model | XGBoost ensemble |
| M5 competition | Combining critical | 220 models but 6 per series |
| FFORMA paper | **14% worse** to use simple average vs learned weights | Same model pool |
| Retail demand study | **19 points MAPE improvement** (9.3% to 7.5%) | Adaptive ensemble |
| Tourism forecasting | **10.2% improvement** over individual models | Prophet + LightGBM + Ridge |

---

## 9. Conformal Prediction for Time Series Uncertainty

### 9.1 What is Conformal Prediction?

Conformal Prediction (CP) is a **post-hoc** uncertainty quantification framework that:
- Works with ANY pre-trained forecasting model
- Provides **distribution-free** prediction intervals
- Guarantees finite-sample coverage (if exchangeability holds)
- Requires only a calibration set

### 9.2 Challenges for Time Series

Standard CP assumes **exchangeability** (i.i.d. data), which is violated in time series due to:
- Temporal dependencies
- Distribution shifts
- Autocorrelation in errors

### 9.3 Best Methods for Time Series (Benchmarked)

A comprehensive 2026 benchmark evaluated CP methods for time series:

| Method | Coverage at 90% | Efficiency (Winkler Score) | Recommendation |
|--------|----------------|---------------------------|----------------|
| **MSCP** (Multi-step Split CP) | Meets threshold | **Best** | Top choice |
| Parametric-PI | Meets threshold | Good | Simple alternative |
| ACI (Adaptive Conformal Inference) | Meets threshold | Good | Non-stationary data |
| Global-CP | Meets threshold | Moderate | Simplest valid method |
| EnbPI | **Fails** coverage | Poor | Not recommended |
| SPCI | **Fails** coverage (worst) | Poor | Not recommended |

**MSCP methodology**: Use horizon-specific calibration scores and quantiles. For each horizon h, collect residuals and compute empirical quantile q_h. Rolling window updates maintain calibration.

### 9.4 Practical Conformal Prediction Pipeline

```python
# 1. Split: Training set + Calibration set (preserve temporal order)
# 2. Train model on training set
# 3. Generate predictions on calibration set
# 4. Compute nonconformity scores: s = |y_true - y_pred|
# 5. For prediction: Interval = [pred - q_(1-alpha), pred + q_(1-alpha)]
# 6. Rolling update: Shift calibration window forward
```

---

## 10. Probabilistic Forecasting & Competition Scoring

### 10.1 Why Intervals Matter in Competitions

Modern forecasting competitions score **probabilistic forecasts**, not just point predictions:

| Competition | Scoring Metric | Quantiles Required |
|-------------|---------------|-------------------|
| M5 | **Weighted Scaled Pinball Loss** | 9 quantiles (0.1 to 0.9) |
| GEFCom2014 | Average Pinball Loss | 99 quantiles (0.01 to 0.99) |
| EEM20 | Average Pinball Loss | 9 quantiles (0.1 to 0.9) |
| General practice | **CRPS** (Continuous Ranked Probability Score) | Full distribution |

### 10.2 Pinball Loss (Quantile Score)

The pinball loss is the standard scoring rule for quantile forecasting:

```
For quantile tau:
  if y_true >= y_pred: loss = tau * (y_true - y_pred)
  if y_true < y_pred:  loss = (1-tau) * (y_pred - y_true)
```

**Properties**:
- Proper scoring rule (incentivizes honest forecasting)
- Measures both **calibration** (correct coverage) and **sharpness** (narrow intervals)
- Integrates to CRPS when averaged over all quantiles
- Used by M5, GEFCom2014, and many energy forecasting competitions

### 10.3 CRPS (Continuous Ranked Probability Score)

```
CRPS(F, y) = integral over all z of (F(z) - 1{z >= y})^2 dz
```

**Key properties**:
- Generalizes MAE to probabilistic forecasts
- Reduces to absolute error for point forecasts
- Strictly proper scoring rule
- Minimized when predicted distribution equals true distribution
- The "L2 distance" between forecast CDF and empirical CDF

### 10.4 Weighted Interval Score (WIS)

Used in COVID-19 forecasting and modern competitions:
```
WIS = (upper - lower) + penalty for actual outside interval
```

Related to the average of pinball losses at the lower and upper quantiles of the prediction interval.

---

## 11. Key Question: Optimal Ensemble for <100 Days of Data

### 11.1 The Core Answer

For demand forecasting with **<100 days of data**, the optimal ensemble is:

> **3 to 5 diverse models, using simple equal weighting or inverse-MAE weighting, with a Ridge+LightGBM hybrid as the core.**

### 11.2 Detailed Recommendations by Constraint

#### For Very Small Data (< 50 days):

| Model | Role | Why |
|-------|------|-----|
| **Ridge Regression** | Trend capture | Linear extrapolation, prevents overfitting |
| **LightGBM (regularized)** | Nonlinear patterns | Strong regularization, small data robust |
| **ETS/Exponential Smoothing** | Baseline seasonality | Requires minimal data, proven for demand |
| **Ensemble method**: Simple average of 3 models | | |

#### For Small Data (50-100 days):

| Model | Role | Why |
|-------|------|-----|
| **Ridge Regression** | Trend component | Linear extrapolation complement |
| **LightGBM** (2 variants) | Core predictors | Recursive + non-recursive |
| **XGBoost or CatBoost** | Diversity | Different tree algorithm |
| **Seasonal Naive / ETS** | Statistical anchor | Prevents overfitting to recent noise |
| **Ensemble**: Inverse-MAE weighted average of 5 models | | |

### 11.3 Why Not More Models?

With <100 days of data:
1. **Estimation error dominates**: Meta-learners overfit with limited calibration data
2. **Diminishing returns**: Research shows 3-4 models capture most ensemble gains
3. **Model instability**: Small data makes individual model performance highly variable
4. **The forecast combination puzzle**: Simple average often beats complex weighting with limited data

### 11.4 Optimal Ensemble Size: Research Evidence

| Source | Finding | Implication |
|--------|---------|-------------|
| Cost of Ensembling study (2025) | "2-3 models sufficient for optimal accuracy" | Small ensembles work |
| Makridakis & Winkler (1983) | Hyperbolic gain curve | Most gains from first 3-4 |
| FFORMA paper | "Maximum increase 1% when removing any method" | Redundancy common |
| Kaggle competitions | 3-5 models typical in top solutions | Practical validation |

### 11.5 Recommended Ensemble Architecture (<100 Days)

```
Layer 1 (Base Models):
  Model 1: Ridge Regression (trend on time_step)
  Model 2: LightGBM - Recursive (all features + lags)
  Model 3: LightGBM - Non-recursive (direct multi-step)
  Model 4: Seasonal Naive + ETS (statistical baseline)
  Model 5: XGBoost or CatBoost (algorithmic diversity)

Layer 2 (Combination):
  Option A: Simple equal average (safest with limited data)
  Option B: Inverse-MAE weighted (if reliable validation set exists)
  Option C: Trimmed mean (if some models occasionally fail)
```

### 11.6 Practical Implementation Checklist

- [ ] **Ensure diversity**: Mix linear, tree-based, and statistical models
- [ ] **Address trend**: Use Ridge regression or LightGBM with `linear_tree=True`
- [ ] **Validate ensemble**: Use temporal cross-validation (walk-forward)
- [ ] **Weight conservatively**: Start with equal weights, only optimize with sufficient data
- [ ] **Monitor error correlations**: Remove models with >0.7 error correlation
- [ ] **Consider conformal prediction**: Add post-hoc intervals with MSCP
- [ ] **Scale features**: Critical for Ridge regression performance
- [ ] **Regularize heavily**: Small data requires aggressive regularization
- [ ] **Use grouped CV**: Preserve temporal structure in validation
- [ ] **Ensemble 3-5 models**: Capture most gains without overfitting

---

## 12. Summary: Actionable Recommendations

### The Winning Formula for Demand Forecasting Competitions

1. **Start with diversity**: Ridge (trend) + LightGBM (nonlinear) + ETS (statistical) is the proven foundation
2. **Add GBDT variants**: Recursive and non-recursive LightGBM for different error patterns
3. **Keep it small**: 3-5 models capture 90%+ of ensemble gains
4. **Weight simply**: Equal weight or inverse-MAE; avoid complex meta-learners with <100 days
5. **Add uncertainty**: Use conformal prediction (MSCP) for reliable intervals
6. **Optimize for competition metric**: Pinball loss if scoring quantiles, CRPS if full distribution
7. **Address trend explicitly**: GBDT cannot extrapolate; Ridge or `linear_tree` required
8. **Cross-learn when possible**: Train on grouped data (store/category/department levels)

### Quick Reference Table

| Aspect | Recommendation | Evidence |
|--------|---------------|----------|
| **Optimal ensemble size** | 3-5 models | Diminishing returns after 3-4 |
| **Best weighting** | Equal or inverse-MAE | Forecast combination puzzle |
| **Core architecture** | Ridge + LightGBM | Trend + nonlinear patterns |
| **Key diversity sources** | Algorithm + target formulation | M5 winner analysis |
| **Improvement from ensembling** | 5-15% over best single | Kaggle competitions |
| **Uncertainty method** | MSCP conformal prediction | 2026 benchmark results |
| **Competition scoring** | Pinball loss optimization | M5, GEFCom2014 |
| **With <100 days** | Simple average, 3-4 models | Estimation error concerns |

---

## References & Sources

1. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." IJF.
2. Makridakis, S., et al. (2022). "M5 accuracy competition: Results, findings, and conclusions." IJF 38:1346-1364.
3. Montero-Manso, P., et al. (2020). "FFORMA: Feature-based Forecast Model Averaging." IJF.
4. Kaggle Learnings Paper (2020). "Learnings from Kaggle's Forecasting Competitions."
5. Elliott, G. & Liao, J. (2025). "Combining Forecasts - On Why Averaging beats Optimal Weights."
6. Claeskens, G., et al. (2016). "The forecast combination puzzle: A simple theoretical explanation."
7. Wang, X., et al. (2023). "The cost of ensembling: is it always worth combining?" (2025)
8. Xu, C. & Xie, Y. (2023). "Conformal prediction for time series."
9. Gneiting, T. & Raftery, A. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation."
10. A demand forecasting system using hybrid ensemble learning (Springer, 2024).
11. "Forecast with Forecasts: Diversity Matters" (2020).
12. Blanc, S.M. & Setzer, T. (2016). "When to choose the simple average in forecast combination." JBR.

---

*Research compiled from 20+ academic papers, competition reports, and industry best practices. Key insight: For demand forecasting with limited data, a small ensemble of 3-5 diverse models using simple weighting outperforms both single models and complex large ensembles.*
