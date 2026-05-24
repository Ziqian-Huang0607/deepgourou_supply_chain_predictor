# SOTA Time Series Forecasting for Supply Chain: 2025-2026 Research Report

## Executive Summary

This report synthesizes findings from 50+ academic papers, competition reports, and industry benchmarks to identify the most advanced demand forecasting methods for supply chain applications as of mid-2026. The key headline: **LightGBM/XGBoost remain the practical production choice for supply chain with <100 days of data**, while **Time Series Foundation Models (Chronos-2, TimesFM, Moirai 2.0, Time-MoE) represent the new frontier** for zero-shot and transfer learning scenarios. Deep learning models (N-BEATS, N-HiTS, TFT, iTransformer) excel with longer histories and rich covariates but generally underperform gradient boosting on small datasets without extensive tuning and data preprocessing.

---

## 1. M5 Competition (2020): What Won and What Improved Since

### 1.1 M5 Competition Winners

The M5 Forecasting Competition (2020), which used Walmart retail sales data across 30,490 hierarchical time series, marked a pivotal moment in forecasting history:

**M5 Accuracy Competition Top 5:**

| Rank | Team | Approach | Key Innovation |
|------|------|----------|----------------|
| 1st | YeonJun Im (YJ_STU) | **LightGBM ensemble** (220 models) | Equal-weighted combination of recursive + non-recursive LightGBM; Tweedie loss; cross-learning |
| 2nd | Anderer | **LightGBM + N-BEATS hybrid** | LightGBM for intermittent low-level series; N-BEATS for continuous high-level series |
| 3rd | Jeon & Seong | **DeepAR** (43 NNs ensemble) | Distribution-sampled training data; cross-learning via autoregressive RNN |
| 4th | (various) | LightGBM variants | Cross-validation on last four 28-day windows |
| 5th | (various) | LightGBM + pooling | Pooled regression with LightGBM |

**M5 Uncertainty Competition Top 3:**

| Rank | Team | Approach |
|------|------|----------|
| 1st | Lainder & Wolfinger | **LightGBM** with CV and data augmentation |
| 2nd | Mamonov et al. | ML + nonparametric distribution estimation |
| 3rd | Nasios & Vogklis | Gradient boosted trees + neural networks |

### 1.2 Key M5 Findings That Still Hold

1. **LightGBM dominated**: All top 50 methods used "cross-learning" (training one model on multiple series). LightGBM was the "standard method of choice" due to fast computation, native categorical handling, minimal preprocessing, and effective Tweedie loss for zero-inflated sales data.

2. **Combination beats individual models**: The winner used a simple equal-weighted average of 6 models. Finding: simple combinations provide both accuracy and robustness.

3. **Cross-learning is essential**: Training on all series simultaneously beat series-by-series approaches, enabled by the hierarchical structure of retail data.

4. **Exogenous variables matter**: Prices, promotions, calendar effects, and special events were critical features. Models ignoring these performed significantly worse.

5. **Effective CV strategies**: Using at least the last four 28-day windows for cross-validation was essential for avoiding overfitting.

### 1.3 Improvements Since M5 (2020-2026)

**Architectural Advances:**
- **2021**: Temporal Fusion Transformer (TFT) introduced interpretable multi-horizon forecasting with variable selection networks
- **2022**: N-HiTS improved on N-BEATS with hierarchical interpolation for long-horizon forecasting
- **2023**: PatchTST introduced patch-based tokenization for Transformers; iTransformer inverted the Transformer paradigm (attention across channels, not time)
- **2024**: TimeMixer introduced decomposable multiscale mixing; TSMixer brought MLP-mixer to time series
- **2025**: **Time Series Foundation Models** emerged as the dominant new paradigm (Chronos-2, TimesFM 2.0, Moirai 2.0, Time-MoE, Sundial)
- **2026**: TabPFN-TS demonstrated that tabular foundation models can outperform specialized TS models; WaveMoE integrated wavelet features with MoE

**Key Methodological Improvements:**
- **Foundation Models**: Pre-trained on 100B+ time points, enabling strong zero-shot forecasting without task-specific training
- **Mixture of Experts (MoE)**: Time-MoE scaled to 2.4B parameters with sparse activation, achieving SOTA precision with manageable inference cost
- **Probabilistic Deep Learning**: Flow-based methods (Sundial) and improved quantile estimation (Chronos-2)
- **Hybrid Boosting**: LLM-Boost and PFN-Boost fuse transformer priors with gradient boosting for best-of-both-worlds performance

---

## 2. Temporal Fusion Transformer (TFT): SOTA Status in 2025-2026

### 2.1 What is TFT

The Temporal Fusion Transformer (Lim et al., 2021) is an attention-based deep learning architecture designed for multi-horizon forecasting with **interpretability**. Key components:

- **Gated Residual Network (GRN)**: Skip connections + gating for efficient information flow
- **Variable Selection Network (VSN)**: Learns which features are most relevant at each time step
- **LSTM encoder-decoder**: Local temporal processing
- **Multi-head attention**: Long-range dependency learning
- **Static covariate encoders**: Handles static features (product category, store location)
- **Future known input handling**: Naturally incorporates future-known covariates (promotions, holidays)

### 2.2 Is TFT Still SOTA in 2025-2026?

**Verdict: TFT remains competitive for covariate-rich forecasting but is no longer the universal SOTA.**

**Where TFT Excels:**
- **Rich covariate environments**: When you have many exogenous variables (promotions, weather, calendar), TFT's variable selection provides interpretability
- **Multi-horizon probabilistic forecasting**: Native quantile output for inventory optimization
- **Interpretability**: Variable importance scores help supply chain managers understand drivers
- **A 2025 pharmacy retail study** found TFT (MAE=104.2, R2=0.986) outperformed XGBoost (MAE=140.7), LSTM (MAE=130.4), and Linear Regression (MAE=130.3)
- **A 2025 wind forecasting study** showed TFT dominated at 12h and 24h horizons where tree models degraded

**Where TFT Falls Short:**
- **A 2024 retail sales comparison** (Comparative Analysis of Modern ML Models) found:
  - LightGBM: Group Revenue WMAPE = 0.069
  - XGBoost: Group Revenue WMAPE = 0.072
  - **TFT: Group Revenue WMAPE = 0.194** (nearly 3x worse)
  - N-HiTS: 0.192, N-BEATS: 0.221 (also much worse)
- **Computational cost**: TFT training time ~118 seconds vs LightGBM (~seconds), XGBoost (~seconds)
- **Small data regimes**: Deep architectures overfit with limited history; need imputation/preprocessing to perform well

### 2.3 TFT Recommendation for Supply Chain

| Scenario | Recommendation |
|----------|---------------|
| <100 days of data, need speed | **LightGBM/XGBoost** - significantly better accuracy and 100x faster |
| >1 year of data, many covariates, need interpretability | **TFT** - competitive with best interpretability |
| Long-horizon forecasting (12+ periods) | **TFT or iTransformer** - transformers maintain accuracy at distance |
| Need probabilistic forecasts for inventory | **TFT, DeepAR, or Chronos-2** - all produce good quantile forecasts |
| Production deployment with speed constraints | **LightGBM** - much faster inference |

---

## 3. N-BEATS, N-HiTS for Short Time Series (~90 Days)

### 3.1 Architecture Overview

**N-BEATS** (Oreshkin et al., 2020):
- Pure deep learning architecture using only fully connected layers
- Block-based with backcast + forecast paths
- Decomposes into trend (polynomial basis) and seasonality (Fourier basis) components
- Interpretable version provides explainable outputs
- **Key limitation**: Each stack must predict the full forecast horizon, making it expensive for long horizons

**N-HiTS** (Challu et al., 2023) - Improved version:
- **Hierarchical interpolation**: Different stacks predict at different time scales
- Uses "expressiveness ratio" to reduce cardinality per stack
- Multi-rate signal sampling and non-linear regression
- **45x faster than Transformer-based models**, 1.26x faster than N-BEATS
- Requires only 54% of N-BEATS parameters
- **14% lower MAE** and **16% lower MSE** than best baselines on average

### 3.2 Performance on Short Time Series

**Critical finding: N-HiTS/N-BEATS require sufficient data to learn multi-scale patterns.**

**Data requirements (from practitioner guidance):**
- Minimum: **2-3 complete cycles of the longest seasonal pattern**
- For daily data with weekly seasonality: ~21-30 days absolute minimum
- For daily data with yearly seasonality: **2-3 years recommended**
- Each hierarchical level needs 10-20x more training samples than input window size

**Empirical Results:**

| Study | Dataset | N-BEATS/N-HiTS Performance |
|-------|---------|---------------------------|
| Retail demand (2024) | Short-term retail | N-BEATS SMAPE=27.85%, N-HiTS=27.87% vs MSTL baseline=69.08% |
| Oil demand forecasting | Monthly, 7 regions | N-HiTS beat XGBoost (167 models vs 1); MAPE <10% for 1-year forecasts |
| Financial forecasting | Daily price data | N-BEATS outperformed N-HiTS and ARIMA on all metrics |
| Wind power (2026) | Short-term | W-HiTS-Attention variant achieved RMSE=55.56, R2=0.9918 |

### 3.3 Recommendation for ~90 Days of Data

**With only ~90 days (roughly 12 weeks) of data:**

| Model | Suitability | Notes |
|-------|------------|-------|
| **LightGBM/XGBoost** | **Excellent** | Best choice for short data; robust, fast, needs minimal data |
| **N-HiTS** | Moderate | Can work with 90 days if weekly seasonality is strong; reduce hierarchical levels |
| **N-BEATS** | Moderate | Similar to N-HiTS; may overfit with limited history |
| **TFT** | Poor-Moderate | Overfits easily; needs imputation/preprocessing to work |
| **ARIMA/ETS** | Good | Strong baseline for short series; often beats DL with <100 observations |
| **Chronos-2 (zero-shot)** | **Good** | Foundation model may generalize well even with limited fine-tuning data |

**Key insight**: A 2024 study ("Walking Back the Data Quantity Assumption") found that **more data does NOT always improve deep learning forecasts**. For some series, the best results came from early iterations with fewer observations. This suggests that with 90 days, a carefully tuned LightGBM or even ARIMA may outperform deep learning.

---

## 4. DeepAR and Probabilistic Models for Supply Chain

### 4.1 DeepAR Overview

DeepAR (Salinas et al., 2020, Amazon Research) is a **probabilistic forecasting method** using autoregressive recurrent neural networks:

- **Architecture**: RNN (LSTM/GRU) with probabilistic output distribution
- **Training**: Cross-learning across all time series simultaneously
- **Output**: Full predictive distribution via Monte Carlo sampling (not just point forecasts)
- **Distributions**: Gaussian, negative binomial, Student's T (for different data types)
- **Key advantage**: Produces calibrated prediction intervals for inventory optimization

**How it works:**
```
h_t = RNN([y_t, x_{t+1}^{(f)}, x^{(s)}], h_{t-1})
theta_t = Linear(h_t)
y_{t+1} ~ P(y_{t+1} | theta_t)
```

### 4.2 DeepAR for Supply Chain: Strengths

1. **Probabilistic outputs**: Critical for safety stock calculation and service level optimization
2. **Cross-learning**: Shares information across products/stores, helping with cold-start items
3. **Handles multiple distributions**: Negative binomial for intermittent demand; Gaussian for continuous
4. **Covariate support**: Naturally incorporates future-known features (promotions, holidays)

### 4.3 DeepAR Limitations

1. **Training complexity**: Slower to train than LightGBM; requires careful hyperparameter tuning
2. **Data hunger**: Performs best with longer histories (>1 year)
3. **RNN limitations**: Sequential computation limits parallelization; can struggle with very long dependencies
4. **Competition results**: In M5, DeepAR-based solutions placed 3rd but were significantly outperformed by LightGBM ensembles

### 4.4 DeepState

DeepState (Rangapuram et al., 2018) combines state-space models with deep learning:
- Learns the parameters of a state-space model using RNNs
- Good for series with clear structural patterns (trend + seasonality)
- More interpretable than pure neural approaches
- Generally less competitive than DeepAR in recent benchmarks

### 4.5 Probabilistic Model Comparison for Supply Chain

| Model | Probabilistic | Speed | Small Data | Intermittent Demand | Best Use Case |
|-------|-------------|-------|-----------|-------------------|---------------|
| **DeepAR** | Yes | Slow | Moderate | Good (NegBinomial) | Medium+ data, need full distribution |
| **TFT** | Yes (quantile) | Medium | Poor | Moderate | Many covariates, need interpretability |
| **Chronos-2** | Yes (sampling) | Medium | **Excellent** (zero-shot) | Good | Limited training data, need quick deployment |
| **N-BEATS-I** | Yes (recent) | Fast | Moderate | Poor | Univariate, need speed + uncertainty |
| **LightGBM+quantile** | Yes | **Fast** | **Excellent** | Good | Production supply chain, speed critical |
| **Croston/SBA** | No | **Fastest** | **Excellent** | **Best** | Baseline for intermittent demand |

### 4.6 Recommendation

For supply chain demand forecasting with probabilistic needs:
- **Production, speed-critical**: LightGBM with quantile regression (10th, 50th, 90th percentiles)
- **Need full distribution, have GPU resources**: DeepAR or Chronos-2
- **Intermittent demand (many zeros)**: Croston's method or SBA as baseline, then add LightGBM
- **Best of both**: Ensemble LightGBM (point) + DeepAR (distribution) weighted by validation performance

---

## 5. XGBoost/LightGBM vs Deep Learning for <100 Days of Data

### 5.1 The Evidence

This is the most critical question for many supply chain practitioners. The evidence is clear:

**Across multiple studies, gradient boosting wins with small-to-medium datasets:**

**Study 1: M5 Competition (2020)**
- LightGBM ensembles placed 1st, 2nd, 4th, 5th
- Deep learning placed 3rd (and required 43 model ensemble)
- LightGBM was ~100x faster to train

**Study 2: Retail Sales Comparison (2025)**
- LightGBM: Group Revenue WMAPE = 0.069
- XGBoost: Group Revenue WMAPE = 0.072
- TFT: Group Revenue WMAPE = 0.194
- N-HiTS: Group Revenue WMAPE = 0.192
- N-BEATS: Group Revenue WMAPE = 0.221

**Study 3: Tabular Data Benchmark (2025-2026, 19 datasets)**
- On small datasets (<1K rows): TabPFN (foundation model) > LightGBM/XGBoost > standard NNs
- On medium datasets (1K-10K rows): TabPFN still leads, but gap narrows
- On large datasets (>10K rows): XGBoost/LightGBM competitive, TabPFN 3 still leads
- **Gradient boosting consistently underperforms foundation models on small data**

**Study 4: Neural Nets vs Boosted Trees (176 datasets)**
- No single algorithm dominates across all datasets
- GBDTs excel on "irregular" datasets (skewed, heavy-tailed)
- NNs can match GBDTs with proper tuning, but require more effort
- **TabPFN excels in small data scenarios** (<3000 samples)

**Study 5: Short-term electricity demand**
- LightGBM: MAPE = 2.10%, RMSE = 19.53
- LSTM: MAPE = 7.05%, RMSE = 56.75
- **LightGBM was ~3x more accurate and ~10x faster**

### 5.2 Why LightGBM Wins with Limited Data

| Factor | LightGBM/XGBoost | Deep Learning |
|--------|-----------------|---------------|
| **Data efficiency** | Excellent - trees partition feature space efficiently | Poor - millions of parameters need many examples |
| **Overfitting risk** | Low - natural regularization via tree depth, leaf constraints | High - needs dropout, early stopping, batch norm |
| **Feature engineering** | Handles raw features well; built-in categorical support | Requires careful normalization, encoding |
| **Training speed** | Seconds to minutes | Minutes to hours |
| **Hyperparameter sensitivity** | Moderate - defaults often work well | High - needs extensive tuning |
| **Cross-learning** | Easy - single model on all series | Harder - needs careful batching |
| **Missing values** | Native handling | Requires imputation |
| **Interpretability** | Good (SHAP, feature importance) | Poor (except TFT) |

### 5.3 When Deep Learning Can Win with Small Data

Deep learning CAN be competitive with <100 days when:

1. **Transfer learning / Foundation models**: Pre-trained on massive corpora, then fine-tuned
   - TabPFN-TS (11M params) outperforms Chronos-Mini and matches Chronos-Large (65x more params)
   - Chronos-2 zero-shot often beats trained LightGBM on small datasets
   
2. **Proper preprocessing**: Data imputation makes a huge difference
   - In the 2025 retail study, neural models improved dramatically with imputed data
   - N-BEATS went from worst to competitive with imputation

3. **Architectural innovations for small data**:
   - N-HiTS with reduced hierarchical levels
   - DLinear (simple linear model) - surprisingly effective
   - PatchTST with short patches

4. **Hybrid approaches**:
   - PFN-Boost: TabPFN predictions + XGBoost on residuals
   - LLM-Boost: LLM predictions + gradient boosting
   - These achieve SOTA across all dataset sizes

### 5.4 Verdict for <100 Days of Supply Chain Data

| Rank | Approach | Expected Performance | Speed |
|------|----------|---------------------|-------|
| 1 | **Chronos-2 / TabPFN-TS (zero-shot/few-shot)** | Best if available | Fast |
| 2 | **LightGBM with temporal features** | Excellent baseline | Very Fast |
| 3 | **XGBoost with temporal features** | Excellent | Very Fast |
| 4 | **ARIMA/ETS/Prophet** | Strong baseline | Fastest |
| 5 | **N-HiTS/N-BEATS (with tuning)** | Moderate | Medium |
| 6 | **TFT** | Poor-moderate (without imputation) | Slow |
| 7 | **iTransformer/PatchTST** | Poor with <100 days | Medium |

---

## 6. Transfer Learning for Time Series Forecasting

### 6.1 The Foundation Model Revolution (2024-2026)

Transfer learning has emerged as the most significant paradigm shift in time series forecasting. The key idea: **pre-train on massive, diverse time series corpora, then apply zero-shot or fine-tune on your specific data**.

### 6.2 Major Time Series Foundation Models (2025-2026)

| Model | Organization | Parameters | Pretraining Data | Architecture | Key Feature |
|-------|-------------|------------|-----------------|--------------|-------------|
| **Chronos-2** | Amazon | 120M | Diverse public + synthetic | T5 MoE | Tokenization + group attention; multivariate support |
| **TimesFM 2.0** | Google | 200M | 100B+ real-world points | Decoder-only Transformer | Patching, frequency-aware encodings |
| **Moirai 2.0** | Salesforce | 14M-311M | 27B observations (9 domains) | Decoder-only, quantile output | Multi-patch projections, any-variate attention |
| **Time-MoE** | Tsinghua/ICLR 2025 | 50M-2.4B | 300B time points (9 domains) | Sparse MoE Transformer | Scales to 2.4B params with efficient inference |
| **Sundial** | Tsinghua/ICML 2025 | Various | 1 trillion time points | Generative, flow-matching | Flow-based probabilistic forecasting |
| **TabPFN-TS** | Freiburg | 11M | Synthetic tabular tasks | Tabular foundation model | In-context learning, no gradient fine-tuning |
| **TiRex** | NX-AI/NeurIPS 2025 | Compact | 47.5M training TS | xLSTM (enhanced LSTM) | State-of-the-art zero-shot, fewer parameters |
| **MOMENT** | Various | Various | Diverse corpus | Masked encoder transformer | General pretraining for multiple TS tasks |

### 6.3 Can You Pre-train on Walmart/Favorita Then Fine-tune?

**Yes, and this is increasingly the recommended approach.**

**Evidence:**

1. **Chronos zero-shot on M5**: Published benchmarks show Chronos outperforms specialized methods on datasets seen during training and produces comparable zero-shot performance on new datasets

2. **Agricultural price study (2025)**: Zero-shot TSFMs "substantially outperformed both traditional methods and deep learning models trained from scratch, with the smallest foundation model (Time-MoE) achieving the best overall performance"

3. **Process model forecasting (2025)**: TSFMs in zero-shot mode "substantially outperformed two selected baselines" and generalization was strong across heterogeneous datasets

4. **N-BEATS pretraining**: The original N-BEATS paper demonstrated pretraining on M4 then transferring to new datasets improved accuracy

**Practical Transfer Learning Workflow:**

```
Step 1: Start with a foundation model (Chronos-2, TimesFM, or Moirai 2.0)
Step 2: Evaluate zero-shot performance on your data
Step 3: If needed, apply parameter-efficient fine-tuning (LoRA, rank=2-4)
Step 4: For best results, combine with LightGBM ensemble
Step 5: Monitor for domain shift; retrain foundation model head as needed
```

### 6.4 Fine-tuning Strategies

| Strategy | Description | When to Use | Performance Gain |
|----------|------------|-------------|-----------------|
| **Zero-shot** | Use pre-trained model directly | Quick deployment, limited data | Baseline - often surprisingly good |
| **LoRA fine-tuning** | Low-rank adaptation of attention weights | Small-medium datasets, avoid overfitting | 5-15% improvement |
| **Full fine-tuning** | Update all parameters | Large dataset, different domain | 10-25% improvement |
| **In-context learning** | Provide examples in context (TabPFN) | Very small datasets | Can match full training |
| **Hybrid: Foundation + GBDT** | Foundation predictions as GBDT features | Best accuracy needed | Often SOTA |

### 6.5 Transfer Learning Limitations

1. **Domain dependency**: Zero-shot capabilities are tied to pretraining domains. A 2025 study found that "fine-tuned foundation models do not consistently outperform smaller dedicated models relative to their increased parameter count"

2. **Limited covariate support**: Most foundation models are univariate; covariate support is improving but still limited (Chronos-2 and Moirai are exceptions)

3. **Computational requirements**: Foundation models need GPU inference; not suitable for all edge deployments

4. **"Context parroting" critique**: Some foundation models simply match patterns in context rather than learning true dynamics, especially on chaotic systems

### 6.6 Recommendation for Supply Chain

**For a new supply chain forecasting project:**
1. **Immediate**: Deploy Chronos-2 or TimesFM in zero-shot mode as your first baseline
2. **Week 1**: Build LightGBM with temporal features as your strong baseline
3. **Week 2**: Fine-tune foundation model with LoRA on your data
4. **Week 3**: Ensemble the two approaches (foundation model + LightGBM)
5. **Ongoing**: As more data accumulates, shift weight toward LightGBM; keep foundation model for new/cold-start items

---

## 7. What's New in 2025-2026: Breakthrough Methods

### 7.1 Time Series Foundation Models (Biggest Shift)

The emergence of TS foundation models trained on 100B+ time points represents the biggest paradigm shift since deep learning entered forecasting:

**Key releases in 2025:**
- **Chronos-2** (Amazon, Oct 2025): MoE architecture, 120M params, extends to multivariate forecasting
- **Sundial** (ICML 2025): Flow-matching for continuous values, pretrained on 1 trillion points
- **Time-MoE** (ICLR 2025): First billion-scale TS foundation model (2.4B params), sparse MoE
- **Moirai 2.0** (2025): Decoder-only with quantile forecasting, top-5 on Gift-Eval benchmark
- **TabPFN-TS** (2025): Tabular foundation model that outperforms specialized TS models

### 7.2 Mixture of Experts (MoE) for Time Series

Time-MoE and Chronos-2 introduced sparse MoE to time series:
- **Time-MoE**: 2.4B total parameters, but only ~50M-200M activated per prediction
- Enables massive model capacity without proportional inference cost
- Different "experts" specialize in different pattern types (trend, seasonality, volatility)
- **Results**: Consistently outperforms dense models with same activated parameters

### 7.3 TabPFN and Tabular Foundation Models

The surprise finding of 2025: **tabular foundation models can beat specialized time series models**:
- TabPFN-TS (11M parameters) outperforms Chronos-Large (710M parameters, 65x larger)
- Pre-trained only on synthetic tabular data - no time series training needed
- Uses simple feature engineering from timestamps
- Performs in-context learning without gradient-based fine-tuning
- **Implication**: For small datasets, TabPFN-TS may be the best starting point

### 7.4 PFN-Boost and LLM-Boost

Novel hybrid approaches combining transformers with gradient boosting:
- **PFN-Boost**: TabPFN predictions + XGBoost on residuals
- **LLM-Boost**: LLM predictions + gradient boosting
- Achieves SOTA across all dataset sizes
- Outperforms both standalone transformers and standalone GBDTs

### 7.5 WaveMoE: Frequency-Domain Foundation Models

WaveMoE (2026) integrates wavelet-based frequency features with time-domain features:
- Dual pathways for time and frequency domain modeling
- Decoder-only MoE architecture
- Outperforms pure time-domain models on periodic data
- Particularly relevant for demand forecasting with strong seasonality

### 7.6 SSM Replacing Attention (Mamba for Time Series)

Selective State Space Models (Mamba) are being adapted for time series:
- **Linear time complexity** vs O(n^2) for attention
- Smaller memory footprint
- Competitive accuracy with transformers
- SSM-CGM (2025) replaced TFT's LSTM + attention with Mamba blocks
- **Implication**: May enable efficient long-sequence forecasting on resource-constrained devices

### 7.7 Quantum-Enhanced TFT

QTFT (2025) explores quantum computing for forecasting:
- Variational quantum algorithm + classical TFT
- Trainable on current NISQ devices
- Early results show improvement over classical TFT on small datasets
- **Status**: Experimental, not yet practical for production

---

## 8. Practical Recommendations for Supply Chain Demand Forecasting

### 8.1 Decision Framework

```
HOW MUCH DATA DO YOU HAVE?
|
|-- < 90 days (< 100 observations per series)
|   |-- Need immediate deployment
|   |   --> Chronos-2 zero-shot OR TabPFN-TS
|   |
|   |-- Can build training pipeline
|   |   --> LightGBM with temporal features (BEST overall)
|   |   --> XGBoost as alternative
|   |   --> Ensemble with ARIMA/Prophet
|
|-- 90 days - 1 year (100-365 observations)
|   |-- Rich covariates (promotions, weather, holidays)
|   |   --> LightGBM + TFT ensemble
|   |
|   |-- Simple univariate series
|   |   --> N-HiTS or N-BEATS
|   |
|   |-- Need probabilistic forecasts
|   |   --> DeepAR or Chronos-2
|
|-- > 1 year (365+ observations)
|   |-- Rich covariates, need interpretability
|   |   --> TFT or iTransformer
|   |
|   |-- Best accuracy, many series
|   |   --> PatchTST or TimeMixer++
|   |
|   |-- Need speed at scale
|   |   --> LightGBM still competitive
|   |
|   |-- Want cutting-edge
|   |   --> Fine-tuned Time-MoE or Moirai 2.0
```

### 8.2 Recommended Model Stack for 2025-2026

For a typical supply chain demand forecasting system, we recommend a **3-tier architecture**:

**Tier 1 - Fast Baseline (always run):**
- LightGBM with Tweedie loss, recursive + direct forecasts
- Features: lags, rolling statistics, calendar, promotions, prices
- Cross-validation: last four 28-day windows

**Tier 2 - Foundation Model (when available):**
- Chronos-2 or TimesFM for zero-shot/few-shot forecasts
- Provides generalization to new items, stores, categories
- Use as feature in Tier 1 ensemble

**Tier 3 - Specialized (for high-value series):**
- TFT for top 20% of SKUs with rich covariates
- DeepAR for series needing full probabilistic distributions
- N-HiTS for long-horizon forecasts

**Final Prediction:** Weighted ensemble of all tiers, optimized by validation performance

### 8.3 Key Hyperparameters

**LightGBM (recommended starting point):**
```python
params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1
}
```

**N-HiTS (for short time series):**
- Reduce `n_pool_kernel_size` to [2, 2] instead of default [2, 2, 2]
- Reduce `n_blocks` per stack to 1-2
- Use `input_size` = 30 (for 90 days of data)
- Use `h` (horizon) = 7 (weekly forecasts)

**TFT (when covariates are rich):**
- `hidden_size`: 128-256
- `attention_head_size`: 4-8
- `dropout`: 0.1
- `learning_rate`: 0.001-0.01
- Early stopping patience: 10

---

## 9. Summary Table: All Models Compared

| Model | Year | Best For | Small Data | Speed | Interpretability | Probabilistic | Key Limitation |
|-------|------|----------|-----------|-------|-----------------|---------------|----------------|
| **LightGBM** | 2017 | Supply chain baseline | Excellent | Very Fast | Good (SHAP) | Via quantile | Needs feature engineering |
| **XGBoost** | 2016 | Alternative to LightGBM | Excellent | Very Fast | Good | Via quantile | Slightly slower than LGBM |
| **N-BEATS** | 2020 | Univariate, interpretable | Moderate | Fast | Excellent (decomposition) | Recent versions | Needs sufficient data cycles |
| **N-HiTS** | 2023 | Long-horizon, efficient | Moderate | Fast | Good | No | Requires multi-scale patterns |
| **TFT** | 2021 | Multi-covariate, interpretable | Poor-Medium | Slow | Excellent | Yes (quantile) | Computationally expensive |
| **DeepAR** | 2020 | Probabilistic, cross-learning | Moderate | Slow | Poor | Yes (sampling) | Slow training, needs GPU |
| **iTransformer** | 2024 | Long-range multivariate | Poor | Medium | Poor | No | Needs long histories |
| **PatchTST** | 2023 | Accurate multivariate | Poor | Medium | Poor | No | Needs tuning |
| **Chronos-2** | 2025 | Zero-shot, quick deploy | **Excellent** | Medium | Poor | Yes | Limited covariate support |
| **TimesFM** | 2024 | Zero-shot point forecasts | **Excellent** | Medium | Poor | No | Deterministic only |
| **Moirai 2.0** | 2025 | Universal probabilistic | **Excellent** | Medium | Poor | Yes (mixture) | Newer, less battle-tested |
| **Time-MoE** | 2025 | Best accuracy at scale | Good | Medium-Fast | Poor | Yes | Large model, complex |
| **TabPFN-TS** | 2025 | Small data, no training | **Best** | Fast | Poor | Yes | Row/feature limits |
| **TimeMixer++** | 2025 | Universal TS analysis | Moderate | Fast | Poor | No | Jack of all trades |

---

## 10. References and Sources

### Academic Papers
1. Makridakis et al. (2022). "The M5 Accuracy competition: Results, findings and conclusions." *International Journal of Forecasting*.
2. Lim et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." *International Journal of Forecasting*.
3. Oreshkin et al. (2020). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." *ICLR*.
4. Challu et al. (2023). "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting." *AAAI*.
5. Salinas et al. (2020). "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks." *International Journal of Forecasting*.
6. Liu et al. (2024). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." *ICLR*.
7. Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR*.
8. Ansari et al. (2024). "Chronos: Learning the Language of Time Series." *TMLR*.
9. Ansari et al. (2025). "Chronos-2: From Univariate to Universal Forecasting."
10. Das et al. (2024). "TimesFM: A decoder-only foundation model for time-series forecasting." *ICML*.
11. Woo et al. (2024). "Unified Training of Universal Time Series Forecasting Transformers." *ICML*.
12. Shi et al. (2025). "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts." *ICLR*.
13. Hoo et al. (2025). "TabPFN-TS: TabPFN Outperforms Specialized Time Series Forecasting Models."
14. Wang et al. (2024). "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting." *ICLR*.
15. Lainder & Wolfinger (2024). "Forecasting with gradient boosted trees: Winning solution to the M5 Uncertainty competition." *IJF*.

### Competition Reports
- M5 Competition Special Issue, *International Journal of Forecasting*
- Kaggle M5 Competition Discussion Forums (2020)

### Key Benchmarks
- Gift-Eval: Comprehensive evaluation benchmark for TS foundation models
- FEV-Bench: Foundation model evaluation benchmark
- M4/M5 Competition datasets
- ETT, Traffic, Weather, Electricity standard benchmarks

---

*Report compiled: July 2026*
*Sources: 50+ academic papers, competition reports, industry benchmarks, and foundation model evaluations*
