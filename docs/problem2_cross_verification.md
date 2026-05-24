# Cross-Verification & Insight Synthesis: Problem 2 Winning Solution

## High Confidence Findings (Confirmed by 3+ dimensions)

### 1. LightGBM + Heavy Feature Engineering is THE Winning Formula
- **Confirmed by**: Dim01 (SOTA architectures), Dim02 (GBDT FE), Dim07 (Ensembles), Dim11 (AutoFE)
- **Evidence**: Every major Kaggle forecasting competition (M5, Rossmann, Corporacion Favorita, Rohlik) was won or dominated by GBDT with extensive feature engineering
- **For our problem**: With only 2 months of data, LightGBM Tweedie loss is optimal for count/demand data

### 2. External Data is the Key Differentiator
- **Confirmed by**: Dim03 (Chinese holidays), Dim04 (Weather), Dim05 (Geographic), Dim06 (Supply chain), Dim12 (Data sources)
- **Evidence**: Organizers explicitly state pure time-series gives poor results. Kaggle winners gained 5-31% from external features
- **Free data sources verified**: Open-Meteo (weather, no API key), Nager.Date (holidays, free), lunardate (Chinese calendar)

### 3. Chinese Calendar Features are Critical
- **Confirmed by**: Dim03 (Holidays), Dim05 (Regional), Dim06 (Supply chain)
- **Evidence**: Lunar New Year 2026 (Feb 17) creates 4-6 week pre-holiday stocking surge. Qingming (Apr 4-6) falls IN prediction period
- **Implementation**: Days until/next holiday, lunar month/day, solar terms, make-up workday flags

### 4. Ensembling is Essential
- **Confirmed by**: Dim01, Dim07, Dim09
- **Evidence**: All Kaggle winners use ensembles. With short data, simple averaging of 3-5 diverse models is optimal
- **Best combo**: LightGBM + XGBoost + Ridge trend (addresses GBDT trend extrapolation weakness)

### 5. Short Rolling Windows Only
- **Confirmed by**: Dim02, Dim09, Dim12
- **Evidence**: Corporacion Favorita winner used only 1-5 months despite 55 months available. With 60 days, use 7 and 14-day windows

## Medium Confidence Findings (1-2 sources, authoritative)

### 6. Probabilistic Forecasting Impresses Judges
- **Source**: Dim08 (Uncertainty quantification)
- **Value**: Even if not scored, conformal prediction intervals demonstrate research depth

### 7. Causal Inference Shows Advanced Understanding
- **Source**: Dim10 (Causal inference)
- **Value**: Google Causal Impact on Lunar New Year effect is intellectually impressive

### 8. Foundation Models as Zero-Shot Baseline
- **Source**: Dim01, Dim09
- **Value**: TimesFM/Chronos can provide strong baseline with zero training data

## Conflict Zones

### Weather Data Value
- **Dim04**: Weather features explain only 3.5-6% additional variance
- **Dim02**: Rossmann winner used weather (precipitation + temperature) successfully
- **Resolution**: Include weather as secondary feature set — not primary, but contributes to ensemble diversity

### Deep Learning vs GBDT for Short Series
- **Dim01**: DeepAR good for cross-learning but complex transformers overfit
- **Dim02**: GBDT consistently wins with proper feature engineering
- **Resolution**: Use LightGBM as primary, add simple DLinear/DeepAR as ensemble member for diversity

## Final Architecture Decision

```
PRIMARY MODEL: LightGBM (Tweedie loss)
  ├── 100+ hand-engineered features
  ├── Chinese calendar features (lunar, holidays, solar terms)
  ├── Weather features (temp, precipitation from Open-Meteo)
  ├── Geographic features (city tier, province)
  ├── Rolling statistics (7d, 14d windows)
  ├── Lag features (1, 7, 14)
  ├── Mean encodings (customer, product, warehouse, dayofweek)
  └── Event counters (days to/from holidays)

SECONDARY MODEL: XGBoost (Tweedie loss)
  └── Same features, different algorithm for diversity

TREND MODEL: Ridge Regression
  └── Addresses GBDT's inability to extrapolate trends

ENSEMBLE: Weighted average (inverse-MAE weights)

INNOVATION (Bonus): 
  ├── Conformal prediction intervals
  ├── Causal impact analysis (Lunar New Year effect)
  └── Automated feature selection (BorutaPy)
```
