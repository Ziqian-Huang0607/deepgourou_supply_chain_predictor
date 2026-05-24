# Dimension 8: Probabilistic Forecasting & Uncertainty Quantification

## Executive Summary

Probabilistic forecasting — predicting the full distribution of possible outcomes rather than a single point — represents one of the most impactful differentiators in supply chain forecasting competitions. While most teams submit only point forecasts, demonstrating uncertainty quantification signals research sophistication and directly addresses the core business need: inventory planning requires not just expected demand, but understanding the *range* of possible demand. The M5 competition introduced formal uncertainty scoring with remarkable success: the winning solution produced "remarkably accurate estimation of prediction uncertainty" using 126 LightGBM quantile models [^5^]. Modern approaches span quantile regression with GBDTs, conformal prediction (the hot new distribution-free technique), MC Dropout for neural networks, and specialized libraries like NGBoost and MAPIE.

---

## 1. Key Findings

### Finding 1: LightGBM Dominates Quantile Regression for Forecasting
Claim: LightGBM is the dominant approach for uncertainty quantification in forecasting competitions, with built-in quantile loss support that enables direct prediction of any quantile [^143^].
Source: MDPI Survey of ML Methods for Time Series Prediction
URL: https://www.mdpi.com/2076-3417/15/11/5957
Date: 2025-05-26
Excerpt: "The Uncertainty competition tasked participants with forecasting the distribution of realized values... Although this competition was less popular, attracting 892 teams, the reliance on LightGBM models remained consistent, with four of the top five submissions incorporating it in their frameworks."
Confidence: High

### Finding 2: M5 Competition Proved Uncertainty Can Be Scored
Claim: The M5 Uncertainty competition required predictions of nine quantiles (0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, and 0.995) using the Weighted Scaled Pinball Loss function, and was won with 126 LightGBM models [^143^].
Source: A Survey of Machine Learning Methods for Time Series Prediction, MDPI
URL: https://www.mdpi.com/2076-3417/15/11/5957
Date: 2025-05-26
Excerpt: "First Place: utilized 126 LightGBM models, one for each quantile and aggregation level... the Uncertainty competition was dominated by Kaggle masters and grandmasters with strong statistical backgrounds."
Confidence: High

### Finding 3: Conformal Prediction Is the Hot New Distribution-Free Technique
Claim: Conformal prediction provides finite-sample coverage guarantees without any assumptions about data distribution or model, making it ideal for time series where traditional assumptions fail [^387^].
Source: Copula Conformal Prediction for Multi-step Time Series Forecasting
URL: https://arxiv.org/pdf/2212.03281v2.pdf
Date: 2022-12
Excerpt: "CP has become popular because of its simplicity, theoretical soundness, and low computational cost. A key difference between CP and other UQ methods is that under the exchangeability assumption, conformal methods guarantee validity in finite samples."
Confidence: High

### Finding 4: EnbPI Extends Conformal Prediction to Time Series
Claim: EnbPI (Ensemble Batch Prediction Intervals) is the first conformal prediction algorithm designed specifically for time series that doesn't require data exchangeability, using bootstrap ensembles and out-of-bag predictions [^432^].
Source: Conformal prediction for time series (Xu & Xie, 2021)
URL: https://arxiv.org/pdf/2010.09107
Date: 2020-10
Excerpt: "EnbPI has a training phase and the prediction phase. In the training phase, EnbPI first fits a fixed number of bootstrap estimators from subsets of the training data. Then, it aggregates predictions from these bootstrap estimators on the training data in an efficient leave-one-out fashion."
Confidence: High

### Finding 5: Quantile Regression Requires Monotonicity Constraints
Claim: When training separate models for each quantile, a well-known "quantile crossing" problem occurs where higher quantiles predict lower values than lower quantiles, requiring post-processing or structural constraints [^495^].
Source: Solution to the Non-Monotonicity and Crossing Problems in Quantile Regression
URL: https://arxiv.org/pdf/2111.04805
Date: 2021-11
Excerpt: "A higher number of points may be captured by a lower quantile line as compared to the higher quantile line. Therefore, the crossing problem is manifested as a non-monotonicity problem as percentiles increase."
Confidence: High

### Finding 6: NGBoost Provides Native Probabilistic Prediction
Claim: NGBoost (Natural Gradient Boosting) outputs full probability distributions rather than point estimates, providing uncertainty quantification natively through natural gradient descent on probability distribution manifolds [^504^].
Source: NGBoost: Natural Gradient Boosting for Probabilistic Prediction (Stanford)
URL: https://ar5iv.labs.arxiv.org/html/1910.03225
Date: 2020
Excerpt: "NGBoost is modular with respect to choice of base learner, distribution, and scoring rule... We estimate the conditional probability distribution P(y|x) for each value of x instead of producing a point estimate."
Confidence: High

### Finding 7: CRPS Is the Gold Standard Evaluation Metric
Claim: The Continuous Ranked Probability Score (CRPS) is the standard metric for evaluating probabilistic forecasts, generalizing MAE to entire distributions and simultaneously evaluating sharpness and calibration [^428^].
Source: Continuous Ranked Probability Score in probabilistic forecasting
URL: https://cienciadedatos.net/documentos/py74-continuous-ranked-probability-score-crps-probabilistic-forecasting
Date: 2026-04-23
Excerpt: "CRPS is widely used in probabilistic forecasting because it provides a unified framework for evaluating both the sharpness (narrowness) and calibration (accuracy) of predictive distributions."
Confidence: High

---

## 2. Major Techniques

### 2.1 Quantile Regression with GBDT (LightGBM/XGBoost)

**How it works:** Instead of predicting the conditional mean E[Y|X], quantile regression predicts conditional quantiles Q_tau(Y|X) by minimizing the pinball loss:

```
L_tau(y, y_hat) = tau * max(y - y_hat, 0) + (1 - tau) * max(y_hat - y, 0)
```

For a 90% prediction interval, train three separate models:
- One for tau=0.5 (median/point forecast)
- One for tau=0.05 (lower bound)
- One for tau=0.95 (upper bound)

**LightGBM Implementation:**

```python
import lightgbm as lgb
import numpy as np

# Shared parameters
params = {
    'objective': 'quantile',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_estimators': 500,
    'random_state': 42
}

# Train separate models for each quantile
quantiles = [0.1, 0.5, 0.9]
models = {}

for q in quantiles:
    model = lgb.LGBMRegressor(**params, alpha=q)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='quantile',
              early_stopping_rounds=50,
              verbose=False)
    models[q] = model

# Predict
pred_10 = models[0.1].predict(X_test)  # Lower bound
pred_50 = models[0.5].predict(X_test)  # Median (point forecast)
pred_90 = models[0.9].predict(X_test)  # Upper bound
```

**Key considerations [^396^]:**
- Training multiple models takes more time
- No guarantee of monotonicity (quantile crossing) — the 10th percentile prediction may sometimes exceed the 50th percentile
- Each quantile may need different hyperparameters
- LightGBM's native `quantile` objective works well out of the box

**XGBoost quantile regression [^434^]:**
XGBoost historically struggled with quantile regression because the pinball loss is not differentiable everywhere. Solutions include:
- QXGBoost: Uses Huber norm function for smooth approximation [^434^]
- Arctan pinball loss: A novel smooth approximation with large second derivative [^495^]
- XGBoost's `reg:quantileerror` objective (newer versions)

### 2.2 Conformal Prediction (CP)

**Standard Split Conformal Prediction:**
1. Split data into training and calibration sets
2. Train model on training set
3. Compute non-conformity scores (absolute residuals) on calibration set
4. The prediction interval is: `[prediction - q_(1-alpha), prediction + q_(1-alpha)]`

**EnbPI for Time Series [^432^]:**
```
1. Train B=20-50 bootstrap models on the full training data
2. For each point, compute out-of-bag (OOB) prediction using only models that didn't see that point
3. Compute OOB residuals for all training points
4. For test prediction: center = mean of all B model predictions
   interval = center +/- quantile of recent residuals
5. Slide residual window forward as new observations arrive
```

**MAPIE Implementation [^425^]:**

```python
from mapie.regression import MapieRegressor, MapieQuantileRegressor
from mapie.subsample import BlockBootstrap
from mapie.time_series_regression import MapieTimeSeriesRegressor
import xgboost as xgb

# --- Standard conformal prediction ---
reg = xgb.XGBRegressor(n_estimators=1000, random_state=42)
reg.fit(X_train, y_train)

mapie = MapieRegressor(reg, method="enbpi", cv="split")
mapie.fit(X_cal, y_cal)
y_pred, y_pis = mapie.predict(X_test, alpha=0.1)  # 90% PI

# --- Time series conformal prediction ---
mapie_cv = BlockBootstrap(n_blocks=10, overlapping=True, random_state=42)
mapie_ts = MapieTimeSeriesRegressor(
    reg, method="enbpi", cv=mapie_cv, agg_function="mean"
)
mapie_ts.fit(X_train, y_train)
y_pred, y_pis = mapie_ts.predict(X_test, alpha=0.05, ensemble=True)

# With online adaptation (partial_fit)
for step in range(gap, len(X_test), gap):
    mapie_ts.partial_fit(
        X_test.iloc[(step - gap):step, :],
        y_test.iloc[(step - gap):step]
    )
    y_pred[step:step + gap], y_pis[step:step + gap, :, :] = \
        mapie_ts.predict(X_test.iloc[step:(step + gap), :], alpha=0.05)
```

**Conformalized Quantile Regression (CQR) [^515^]:**
Combines quantile regression with conformal prediction to guarantee coverage:
1. Train lower and upper quantile regressors (e.g., tau=0.1 and tau=0.9)
2. Compute calibration scores: `s_i = max(q_lower(x_i) - y_i, y_i - q_upper(x_i))`
3. Adjust intervals by quantile of calibration scores

Results show CQR achieves near-exact coverage (~91% for target 90%) across LightGBM, RNN-GRU, and TFT models [^515^].

### 2.3 MC Dropout for Deep Learning Uncertainty

**How it works [^383^]:** Gal & Ghahramani (2016) showed that applying dropout at inference time approximates Bayesian inference. By running T forward passes with dropout enabled, we get T different predictions whose variance quantifies uncertainty.

```python
import torch
import torch.nn as nn

class MCDropoutLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Dropout at inference!
        return self.fc(out)
    
    def predict_with_uncertainty(self, x, n_samples=1000):
        self.train()  # Keep dropout active!
        predictions = torch.stack([self.forward(x) for _ in range(n_samples)])
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std

# Usage
model = MCDropoutLSTM(input_size=10, hidden_size=64, output_size=1)
model.load_state_dict(torch.load('model.pt'))
mean, std = model.predict_with_uncertainty(X_test, n_samples=20000)
```

**Key parameters [^397^]:**
- Number of forward passes: 10,000-20,000 for stable estimates
- Dropout rate: 0.1-0.2 typically
- Captures both aleatoric (data noise) and epistemic (model uncertainty) uncertainty
- ~49 seconds for 20,000 passes on AirPassengers dataset (consumer laptop)

### 2.4 Deep Ensembles

**How it works:** Train M independent neural networks with different random initializations. The ensemble mean is the point forecast; the ensemble variance quantifies uncertainty.

**Advantages over MC Dropout [^415^]:**
- More reliable uncertainty estimates
- Retains low wall-clock time despite higher total epoch count
- No modification to model architecture needed

**Trade-off:** Need to train M separate models (typically M=5-10).

### 2.5 NGBoost for Probabilistic Forecasting

**How it works [^504^]:** NGBoost uses natural gradient boosting to directly learn probability distributions. Instead of predicting a point estimate, it predicts the parameters of a chosen distribution (Normal, Poisson, Gamma, etc.).

```python
from ngboost import NGBRegressor
from ngboost.distns import Normal

# NGBoost outputs full distributions
ngb = NGBRegressor(Dist=Normal, verbose=False)
ngb.fit(X_train, y_train)

# Predict distribution parameters (mean and variance for Normal)
y_dist = ngb.pred_dist(X_test)
mean = y_dist.mean()       # Point predictions
std = y_dist.std()         # Uncertainty

# Get prediction intervals directly
lower = y_dist.ppf(0.05)   # 5th percentile
upper = y_dist.ppf(0.95)   # 95th percentile
```

**Key advantages:**
- Native uncertainty quantification — no post-processing
- Multiple distribution families supported
- Uses CRPS or log-score as scoring rules
- Natural gradient ensures stable optimization on probability manifolds

### 2.6 Bootstrap Prediction Intervals

**How it works [^417^]:** Assume future errors will resemble past errors. Sample with replacement from historical forecast errors and add to point predictions to generate possible future paths.

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import numpy as np

# Fit point forecast model
forecaster = ForecasterAutoreg(regressor=LGBMRegressor(), lags=24)
forecaster.fit(y=y_train)

# Generate bootstrapped predictions
predictions = forecaster.predict_bootstrapping(
    steps=12,
    n_bootstaps=500,  # Number of sample paths
    in_sample_residuals=True
)

# Extract quantiles for prediction intervals
pred_quantiles = predictions.quantile(q=[0.025, 0.5, 0.975], axis=1)
```

### 2.7 Foundation Models (Chronos) for Probabilistic Forecasting

**Amazon Chronos [^141^]:** A pretrained transformer model that tokenizes time series and generates probabilistic forecasts by sampling multiple future trajectories autoregressively.

```python
from autogluon.timeseries import TimeSeriesPredictor

# Zero-shot probabilistic forecasting
predictor = TimeSeriesPredictor(
    prediction_length=24,
    presets="chronos_bolt_base"
).fit(train_data=train_data)

# Returns multiple quantiles by default
predictions = predictor.predict(test_data)
# Contains quantiles: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
```

---

## 3. Implementation Details

### 3.1 Pinball Loss (Quantile Loss)

The pinball loss for quantile level tau is [^484^]:
```
L_tau(y, y_hat) = { tau * (y - y_hat)       if y >= y_hat
                  { (1 - tau) * (y_hat - y)   if y < y_hat
```

**Properties:**
- Asymmetric: penalizes under-prediction more for high quantiles, over-prediction more for low quantiles
- For tau=0.5, reduces to MAE
- Convex but non-differentiable at zero
- Requires smoothing for second-order optimization methods

### 3.2 CRPS (Continuous Ranked Probability Score)

CRPS generalizes MAE to probabilistic forecasts [^428^]:
```
CRPS(F, y) = integral from -inf to +inf of (F(z) - I(z >= y))^2 dz
```

**Python implementations:**
```python
from skforecast.metrics import crps_from_predictions, crps_from_quantiles
import properscoring as ps

# From sample predictions (bootstrap samples)
crps = crps_from_predictions(y_true, y_pred_samples)  # y_pred_samples: array of predictions

# From quantiles
crps = crps_from_quantiles(y_true, pred_quantiles, quantile_levels)

# Using properscoring library
crps = ps.crps_ensemble(y_true, y_pred_samples)
```

**Key insight:** CRPS simultaneously evaluates:
- **Calibration:** Do predicted intervals cover the true value at the nominal rate?
- **Sharpness:** How narrow are the prediction intervals?
- Lower CRPS = better probabilistic forecast
- When the forecast collapses to a point, CRPS = MAE

### 3.3 CRPS as Training Objective

Some frameworks directly optimize CRPS during training [^433^]:
```
Loss = sum over all dimensions of CRPS(Y_predicted, y_true)
```

This ensures the model learns to produce calibrated, sharp distributions aligned with the evaluation metric.

### 3.4 Handling the Quantile Crossing Problem

When training separate models per quantile, enforce monotonicity [^498^]:

```python
def enforce_monotonicity(quantile_preds):
    """
    Post-process to ensure quantile monotonicity.
    quantile_preds: dict mapping quantile_level -> predictions array
    """
    quantiles = sorted(quantile_preds.keys())
    preds = np.column_stack([quantile_preds[q] for q in quantiles])
    
    # Sort each row to enforce monotonicity
    preds = np.sort(preds, axis=1)
    
    return {q: preds[:, i] for i, q in enumerate(quantiles)}
```

**Advanced solution:** Use MCQRNN (Monotonic Composite Quantile Regression Neural Network) which applies structural constraints to guarantee monotonicity [^501^], or the exponential sum mapping approach [^498^].

---

## 4. What Works

### 4.1 LightGBM Quantile Regression Is Battle-Tested
- M5 Uncertainty competition was won with 126 LightGBM models [^143^]
- Four of top five M5 Uncertainty submissions used LightGBM
- Native quantile objective requires no custom implementation
- Can be combined with conformal prediction for guaranteed coverage [^515^]

### 4.2 Conformal Prediction + Quantile Regression (CQR) Works Best
- CQR achieves near-exact coverage (~91% for 90% target) across multiple model types [^515^]
- EnbPI adapts well to non-stationary time series without retraining [^432^]
- MAPIE provides easy-to-use Python implementations for both standard and time-series conformal prediction [^425^]

### 4.3 Deep Ensembles Outperform Single Models
- Ensemble of 10 models provides significantly more reliable uncertainty than any single model [^415^]
- Low computational overhead since individual models train quickly
- No architectural changes needed — just different random seeds

### 4.4 EnbPI Is the Go-To for Time Series Conformal Prediction
- Does not require data exchangeability [^437^]
- Uses out-of-bag predictions for efficient leave-one-out estimation
- Sliding residual window adapts to distribution shifts
- Choosing B=20-50 bootstrap models is sufficient [^432^]

### 4.5 NGBoost for End-to-End Probabilistic Prediction
- Single model outputs full probability distributions [^504^]
- No need for separate quantile models or post-processing
- Natural gradient ensures stable training
- Multiple distribution families (Normal, Poisson, Gamma, Laplace)

### 4.6 Probabilistic Forecasting Directly Supports Inventory Decisions
- Safety stock calculation requires quantiles, not just point forecasts
- P(demand <= Q_tau) = tau directly gives the service level
- ICA Sweden reduced safety stock by 32% using AI-driven probabilistic forecasting [^435^]

---

## 5. What Doesn't Work

### 5.1 Naive Quantile Regression Without Calibration
- Raw quantile regression intervals often have poor coverage (e.g., 56% coverage for 80% target) [^512^]
- Each quantile model needs separate hyperparameter tuning
- Quantile crossing degrades prediction quality
- **Fix:** Combine with conformal prediction or use calibration

### 5.2 Assuming Gaussian Errors
- Traditional prediction intervals (mean +/- 2*std) assume normality
- Demand data is often right-skewed, zero-inflated, and non-Gaussian
- Nonparametric approaches (quantile regression, bootstrap) are preferred [^388^]

### 5.3 Single Model for All Quantiles in GBDT
- XGBoost cannot natively predict multiple quantiles with a single model due to pinball loss non-smoothness [^495^]
- Smooth approximations (arctan pinball) help but crossings still occur
- **Fix:** Use separate models per quantile with post-processing, or use NGBoost

### 5.4 Standard Conformal Prediction for Non-Stationary Series
- Standard split conformal prediction fails under distribution shifts (coverage drops to ~84% for 90% target) [^427^]
- Block bootstrap versions (SCP-block) perform even worse
- **Fix:** Use EnbPI or ACI (Adaptive Conformal Inference) for time series

### 5.5 MC Dropout as the Sole Uncertainty Method
- MC Dropout can underestimate uncertainty compared to deep ensembles
- Requires many forward passes (10,000-20,000) for stable estimates
- Theoretical justification assumes specific dropout architectures

---

## 6. Competition Applications

### 6.1 M5 Uncertainty Competition

**Task:** Predict nine quantiles (0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995) for 40,000+ daily Walmart time series.

**Scoring:** Weighted Scaled Pinball Loss

**Top solutions [^143^]:**
- **1st place:** 126 LightGBM models, one per quantile and aggregation level
- **2nd place:** Recursive LightGBM + statistical methods
- **3rd place:** LightGBM + neural network hybrid
- **4th place:** Two LSTM-based neural networks
- **5th place:** 280 LightGBM models in comprehensive ensemble

**Key insight:** The uncertainty competition was "dominated by Kaggle masters and grandmasters with strong statistical backgrounds," unlike the accuracy competition which was won by an undergraduate student [^143^].

### 6.2 Kaggle Forecasting Competitions

Key learnings from Kaggle forecasting competitions [^5^]:
- "GBDT using feature engineering... and neural networks will both perform well... and outperform existing time series benchmarks in terms of both accuracy and uncertainty"
- "To provide prediction intervals, GBDT and neural networks will be adapted by using custom loss functions such as quantile loss"
- "Ensembles of methods will continue to take up the top slots"
- "Hold-out datasets or time series cross-validation will be used by top placers"

### 6.3 Why Uncertainty Matters Even Without Explicit Scoring

Even if the competition only scores point predictions:
1. **Post-hoc analysis:** Show prediction intervals alongside point forecasts
2. **Model selection:** Choose models that are well-calibrated
3. **Decision support:** Recommend inventory levels at different service levels
4. **Demonstrates sophistication:** Shows research depth beyond basic forecasting
5. **Risk analysis:** Identify SKUs/periods with highest forecast uncertainty

---

## 7. Recommended Approach

### 7.1 Primary: LightGBM Quantile Regression + Conformal Prediction

**Why:** Proven in M5, easy to implement, fast training, excellent accuracy.

```python
import lightgbm as lgb
import numpy as np
from mapie.regression import MapieQuantileRegressor

# Step 1: Train quantile LightGBM models
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
models = {}

for q in quantiles:
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    models[q] = model

# Step 2: Predict all quantiles
predictions = {q: models[q].predict(X_test) for q in quantiles}

# Step 3: Enforce monotonicity
predictions = enforce_monotonicity(predictions)

# Step 4: Apply conformal calibration for guaranteed coverage
mapie = MapieQuantileRegressor(
    models[0.5], method="quantile", cv="split", alpha=0.1
)
mapie.fit(X_cal, y_cal)
y_pred, y_pis = mapie.predict(X_test)
```

### 7.2 Secondary: NGBoost for End-to-End Probabilistic Forecasting

**Why:** Native distribution output, no separate models needed, theoretically principled.

```python
from ngboost import NGBRegressor
from ngboost.distns import Normal, Poisson  # Choose based on data type

# Single model outputs full distribution
ngb = NGBRegressor(
    Dist=Normal,
    n_estimators=1000,
    learning_rate=0.01,
    verbose=False
)
ngb.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=50)

# Full probabilistic predictions
dist = ngb.pred_dist(X_test)
median = dist.median()  # For point forecast submission
lower_95 = dist.ppf(0.025)
upper_95 = dist.ppf(0.975)
```

### 7.3 Neural Network Option: DeepAR or TFT

**Why:** If using deep learning for point forecasts, these provide probabilistic output natively.

**DeepAR** (Amazon): RNN-based, outputs distribution parameters, trained by maximizing likelihood [^491^].

**TFT** (Google): Variable selection networks, multi-horizon quantile prediction, interpretable attention [^498^].

### 7.4 Quick Win: Add Bootstrap Intervals to Any Point Forecaster

If you already have a point forecast model, just add:

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg

forecaster = ForecasterAutoreg(regressor=your_model, lags=24)
forecaster.fit(y=y_train)

# Generate prediction intervals via bootstrapping
boot_predictions = forecaster.predict_bootstrapping(
    steps=forecast_horizon, n_bootstaps=500
)

# Extract quantiles
pi_80 = boot_predictions.quantile(q=[0.1, 0.9], axis=1)
pi_95 = boot_predictions.quantile(q=[0.025, 0.975], axis=1)
```

### 7.5 Recommended Workflow for Competition

1. **Primary submission:** Point forecast from best model (median quantile)
2. **Uncertainty analysis:** Provide 80% and 95% prediction intervals
3. **Calibration check:** Report empirical coverage on validation set
4. **Decision support table:** Map quantiles to inventory recommendations
   - Q_0.5: Baseline order quantity
   - Q_0.9: High-service-level order quantity (avoid stockouts)
   - Q_0.95: Maximum reasonable order quantity
5. **Visualization:** Plot point forecast with prediction intervals for key SKUs
6. **Risk analysis:** Identify periods/SKUs with widest intervals (highest uncertainty)

---

## 8. Sources

1. [^5^] "Learnings from Kaggle's Forecasting Competitions" — https://arxiv.org/pdf/2009.07701
2. [^143^] "A Survey of Machine Learning Methods for Time Series Prediction" — https://www.mdpi.com/2076-3417/15/11/5957
3. [^383^] "Uncertainty estimation for time series classification" — https://arxiv.org/html/2412.10528v1
4. [^387^] "Copula Conformal Prediction for Multi-step Time Series Forecasting" — https://arxiv.org/pdf/2212.03281v2.pdf
5. [^390^] "Probabilistic Forecasting in Supply Chain Planning Explained" — https://www.toolsgroup.com/blog/probabilistic-forecasting-in-supply-chain-planning-explained/
6. [^391^] "Bayesian LSTM with Monte Carlo Dropout for Uncertainty-Aware Time Series Forecasting" — https://github.com/patichandanareddy/bayesian-lstm
7. [^393^] "Reasons Why Supply Chain Leaders Are Shifting to Probabilistic Forecasting" — https://johngalt.com/learn/blog/3-reasons-why-supply-chain-leaders-are-shifting-to-probabilistic-forecasting
8. [^394^] "Quantile loss function in lightGBM: unexpected values for prediction intervals" — https://stats.stackexchange.com/questions/667877
9. [^396^] "LightGBM for Quantile Regression" — https://www.geeksforgeeks.org/machine-learning/lightgbm-for-quantile-regression/
10. [^397^] "Monte Carlo Dropout Neural Networks for Forecasting Sinusoidal Time Series" — https://www.mdpi.com/2076-3417/15/8/4363
11. [^412^] "Conformal Predictions with Random Forest in Python" — https://arxiv.org/html/2501.14570v1
13. [^415^] "Bayesian Neural Networks versus deep ensembles" — https://arxiv.org/html/2509.19180v1
14. [^417^] "A Novel Hybrid Approach to Contraceptive Demand Forecasting" — https://arxiv.org/pdf/2502.09685
15. [^418^] "Robust bootstrap prediction intervals for autoregressive time series models" — https://arxiv.org/pdf/2011.07664
16. [^425^] "Conformal Prediction - A Practical Guide with MAPIE" — https://algotrading101.com/learn/conformal-prediction-guide/
17. [^427^] "A Gentle Introduction to Conformal Time Series Forecasting" — https://arxiv.org/html/2511.13608v1
18. [^428^] "Continuous Ranked Probability Score (CRPS) in probabilistic forecasting" — https://cienciadedatos.net/documentos/py74-continuous-ranked-probability-score-crps-probabilistic-forecasting
19. [^432^] "Conformal prediction for time series" — https://arxiv.org/pdf/2010.09107
20. [^433^] "End-to-End Probabilistic Framework for Learning with Hard Constraints" — https://www.arxiv.org/pdf/2506.07003
21. [^434^] "Quantile extreme gradient boosting for uncertainty" — https://arxiv.org/pdf/2304.11732
22. [^435^] "Optimize safety stock levels with next-level planning software" — https://www.relexsolutions.com/resources/safety-stock/
23. [^436^] "Essential Guide to CRPS for Forecasting" — https://towardsdatascience.com/essential-guide-to-continuous-ranked-probability-score-crps-for-forecasting-ac0a55dcb30d/
24. [^437^] "Uncertainty Quantification in Time Series Forecasting" — https://towardsdatascience.com/uncertainty-quantification-in-time-series-forecasting-c9599d15b08b/
25. [^438^] "EnbPI: 时间序列预测不确定性量化的新方法" — https://www.xinfinite.net/t/topic/9364
26. [^439^] "EnbPI方法与应用研究" — https://blog.csdn.net/tmb8z9vdm66wh68vx1/article/details/144732394
27. [^482^] "Achieving Risk Control in Online Learning Settings" — https://arxiv.org/pdf/2205.09095
28. [^484^] "Conditional Quantile Estimation for Uncertain Watch Time" — https://web3.arxiv.org/pdf/2407.12223v3
29. [^490^] "Composite Quantile Regression With XGBoost Using Arctan Pinball Loss" — https://arxiv.org/html/2406.02293v1
30. [^491^] "A DeepAR-Based Modeling Framework for Probabilistic Streamflow Prediction" — https://www.mdpi.com/2073-4441/17/17/2506
31. [^495^] "Solution to the Non-Monotonicity and Crossing Problems in Quantile Regression" — https://arxiv.org/pdf/2111.04805
32. [^498^] "Short-Term Wind Power Non-Crossing Quantile Forecasting" — https://www.mdpi.com/2227-9717/14/8/1310
33. [^501^] "A Hybrid EMD-LASSO-MCQRNN-KDE Framework for Probabilistic Electric Load Forecasting" — https://www.mdpi.com/2227-9717/13/12/3781
34. [^503^] "A general framework for multi-step ahead adaptive conformal heteroscedastic time series forecasting" — https://arxiv.org/pdf/2207.14219.pdf
34. [^504^] "NGBoost: Natural Gradient Boosting for Probabilistic Prediction" — https://ar5iv.labs.arxiv.org/html/1910.03225
35. [^506^] "CNN + BiLSTM with conformal prediction for demand forecasting" — https://ijsrcseit.com/index.php/home/article/download/CSEIT25112857/
36. [^509^] "LightGBM for Quantile Regression" — https://towardsdatascience.com/lightgbm-for-quantile-regression-4288d0bb23fd/
37. [^512^] "Prediction intervals when forecasting with machine learning models" — https://cienciadedatos.net/documentos/py42-forecasting-prediction-intervals-machine-learning
38. [^513^] "NGBoost paper (ICML 2020)" — http://proceedings.mlr.press/v119/duan20a/duan20a.pdf
39. [^515^] "Conformal Prediction for Time Series with CQR" — https://diposit.ub.edu/server/api/core/bitstreams/6941c1c1-fb62-4489-88e7-5b2e0b310958/content
40. [^141^] "Chronos: Learning the Language of Time Series" — https://arxiv.org/html/2403.07815v1

---

## 9. Quick Reference: Code Templates

### Template A: Complete LightGBM Quantile + Conformal Pipeline

```python
"""
Complete probabilistic forecasting pipeline with LightGBM + MAPIE
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from mapie.regression import MapieQuantileRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score

# 1. Prepare data
# X_train, X_val, X_cal, X_test: feature matrices
# y_train, y_val, y_cal, y_test: target vectors

# 2. Define quantiles for full distribution
quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# 3. Train one LightGBM model per quantile
models = {}
for q in quantiles:
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='quantile',
        early_stopping_rounds=50,
        verbose=False
    )
    models[q] = model

# 4. Generate predictions
predictions = {q: models[q].predict(X_test) for q in quantiles}

# 5. Enforce monotonicity (prevent quantile crossing)
def enforce_monotonicity(preds_dict):
    quantiles_sorted = sorted(preds_dict.keys())
    preds_array = np.column_stack([preds_dict[q] for q in quantiles_sorted])
    preds_array = np.maximum.accumulate(preds_array, axis=1)
    return {q: preds_array[:, i] for i, q in enumerate(quantiles_sorted)}

predictions = enforce_monotonicity(predictions)

# 6. Apply conformal prediction for 90% intervals
mapie = MapieQuantileRegressor(
    models[0.5],
    method="quantile",
    cv="split",
    alpha=0.1
)
mapie.fit(X_cal, y_cal)
y_pred, y_pis = mapie.predict(X_test)

# 7. Evaluate
from skforecast.metrics import crps_from_quantiles

crps_scores = []
for i in range(len(y_test)):
    pred_quantiles = np.array([predictions[q][i] for q in quantiles])
    q_levels = np.array(quantiles)
    crps_scores.append(crps_from_quantiles(y_test.iloc[i], pred_quantiles, q_levels))

print(f"Mean CRPS: {np.mean(crps_scores):.4f}")
print(f"90% PI Coverage: {regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]):.4f}")
print(f"90% PI Mean Width: {regression_mean_width_score(y_pis[:, 0, 0], y_pis[:, 1, 0]):.4f}")
```

### Template B: NGBoost Quick Start

```python
"""
Probabilistic forecasting with NGBoost
"""
from ngboost import NGBRegressor
from ngboost.distns import Normal, Poisson
from skforecast.metrics import crps_from_predictions

# Choose distribution based on data type
# Normal for continuous, Poisson for count data
ngb = NGBRegressor(
    Dist=Normal,
    n_estimators=2000,
    learning_rate=0.01,
    minibatch_frac=0.5,
    verbose=False
)

# Fit with early stopping
ngb.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=100)

# Predict full distribution
dist = ngb.pred_dist(X_test)

# Extract any needed statistics
point_forecast = dist.mean()           # For competition submission
median_forecast = dist.median()        # Robust point estimate
uncertainty = dist.std()               # Prediction uncertainty
q_05 = dist.ppf(0.05)                  # Lower 5%
q_95 = dist.ppf(0.95)                  # Upper 95%

# Evaluate with CRPS using samples
samples = dist.sample(1000).squeeze()
crps = np.mean([crps_from_predictions(y_true[i], samples[i]) 
                for i in range(len(y_true))])
```

### Template C: Bootstrap Intervals for Any Forecaster

```python
"""
Add prediction intervals to any trained forecaster via bootstrapping
"""
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import numpy as np

# Wrap your model
forecaster = ForecasterAutoreg(regressor=best_model, lags=24)
forecaster.fit(y=y_train)

# Generate bootstrapped paths
boot_preds = forecaster.predict_bootstrapping(
    steps=forecast_horizon,
    n_bootstaps=1000,
    in_sample_residuals=True,
    random_state=42
)

# Extract prediction intervals
pi_50 = boot_preds.quantile(q=[0.25, 0.75], axis=1)  # 50% PI
pi_80 = boot_preds.quantile(q=[0.1, 0.9], axis=1)    # 80% PI
pi_95 = boot_preds.quantile(q=[0.025, 0.975], axis=1) # 95% PI

median_forecast = boot_preds.quantile(q=0.5, axis=1)
```

---

*Research compiled: July 2025*
*Total sources consulted: 40+*
*Search queries executed: 18*
