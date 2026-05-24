# Dimension 10: Causal Inference for Demand Prediction

## Research Overview

This research investigates causal inference methods as an advanced technique for demand prediction, moving beyond correlation-based forecasting to answer counterfactual questions: "What happens to demand IF we change X?" The research covers seven major causal inference approaches with implementation details, evidence-based best practices, and practical recommendations for short-horizon demand forecasting.

---

## 1. Key Findings

### Finding 1: Causal inference provides actionable business insights that pure forecasting cannot

```
Claim: Amazon's Graph Causal Forecasting (GCF) model achieved 75.3% lower MAPE and 61.2% improvement in recommendation quality by estimating counterfactual demand using a synthetic control approach [^1^]
Source: AAAI Conference Proceedings (2025)
URL: https://ojs.aaai.org/index.php/AAAI/article/view/35148
Date: 2025
Excerpt: "We propose a novel Graph Causal Forecasting (GCF) model, that predicts the unobserved demand leveraging the relationship of a product with other similar products in the marketplace (spatial aspect), along with change in demand over time for each product (temporal aspect)...Using GCF for our demand forecasting problem, we achieve 75.3% lower MAPE compared to baseline."
Confidence: High
```

### Finding 2: DoWhy's four-step workflow (Model, Identify, Estimate, Refute) is the gold standard for rigorous causal analysis

```
Claim: DoWhy uniquely separates the causal problem (identification) from the statistical problem (estimation) and provides automated robustness checks that most libraries skip [^2^]
Source: Microsoft Research (DoWhy paper)
URL: https://arxiv.org/pdf/2011.04216
Date: 2020
Excerpt: "In addition to efficient statistical estimators of a treatment's effect, successful application of causal inference requires specifying assumptions about the mechanisms underlying observed data and testing whether they are valid, and to what extent. However, most libraries for causal inference focus only on the task of providing powerful statistical estimators."
Confidence: High
```

### Finding 3: Causal discovery from 2 months of data is theoretically problematic but practically possible with domain constraints

```
Claim: Reliable causal structure learning typically requires n >> d (samples much greater than variables), with O(d^3) complexity limiting scalability, but background knowledge and hybrid approaches can partially compensate [^3^]
Source: Comprehensive Review and Empirical Evaluation of Causal Discovery Algorithms
URL: https://arxiv.org/html/2407.13054v2
Date: 2024
Excerpt: "Sample Size Requirements: Reliable causal structure learning typically requires n >> d, limiting applicability to high-dimensional, small-sample problems"
Confidence: High
```

### Finding 4: Causal Impact analysis requires at least 3:1 pre-to-post data ratio for reliable results

```
Claim: For causal impact analysis, maintain at least a 3:1 ratio of pre-intervention to post-intervention observations; 12 weeks of pre-campaign data needed to measure a 4-week campaign [^4^]
Source: MCP Analytics - Causal Impact Practical Guide
URL: https://mcpanalytics.ai/articles/causal-impact-practical-guide-for-data-driven-decisions
Date: 2025
Excerpt: "Maintain at least a 3:1 ratio of pre-intervention to post-intervention observations. If you want to measure a 4-week campaign effect, you need at least 12 weeks of pre-campaign data."
Confidence: High
```

### Finding 5: Weather serves as a valid instrumental variable for demand estimation because it is exogenous and affects demand through identifiable channels

```
Claim: Weather has been used as an instrumental variable since 1928 (Wright) to identify demand functions because it is exogenous and acts like a natural experiment [^5^]
Source: NBER Working Paper / Review of Environmental Economics and Policy
URL: https://www.nber.org/system/files/working_papers/w19087/w19087.pdf
Date: 2013
Excerpt: "Wright (1928) used weather as an instrumental variable to identify a demand function for oils. Because weather is exogenous and random in most economic applications, it acts like a 'natural experiment' and thus in some settings allows researchers to identify statistically the causal effect of one variable on an economic outcome of interest."
Confidence: High
```

### Finding 6: Generalized Causal Forest (GCF) outperforms traditional causal forest for continuous treatments like pricing

```
Claim: GCF achieved 15.1% to 25.2% improvement in finished orders over causal forest in online A/B testing at a major ride-sharing company, deployed on Spark [^6^]
Source: KDD 2022 Workshop on Decision Intelligence
URL: https://arxiv.org/abs/2203.10975
Date: 2022
Excerpt: "Compared to CF, GCF improves FO by 15.1% and 25.2% in the single mobility options strategy and dual mobility options strategy, respectively...We implement GCF on Spark and successfully deploy it into a large-scale online pricing system."
Confidence: High
```

### Finding 7: Causal inference quantifies promotion cannibalization effects in grocery retail

```
Claim: Google Causal Impact was successfully used to measure cannibalization effects during promotional sales by analyzing each cannibal-victim product pair as a treatment-control relationship [^7^]
Source: IEEE Access (2021)
URL: https://doi.org/10.1109/ACCESS.2021.3062222
Date: 2021
Excerpt: "We propose to use causal inference to measure the impact of cannibalisation due to promotions. Our method reviews each product that has been on promotion, searching for potential cannibals...each cannibal-victim pair is analysed with Causal Impact."
Confidence: High
```

### Finding 8: Double Machine Learning (DML) from EconML specifically supports cross-price elasticity matrix estimation

```
Claim: EconML's LinearDML can estimate the full matrix of cross-price elasticities by setting Y as demand vector and T as price vector [^8^]
Source: EconML Official Documentation
URL: https://econml.azurewebsites.net/spec/estimation/dml.html
Date: Current
Excerpt: "In settings like demand estimation, we might want to fit the demand of multiple products as a function of the price of each one of them...a_hat[i,j] contains the elasticity of the demand of product i on the price of product j"
Confidence: High
```

---

## 2. Major Techniques

### 2.1 CausalML / Uplift Modeling

**Concept**: Uplift modeling estimates the incremental effect of a treatment (promotion, price change, feature display) on an individual unit's demand, rather than just predicting who will buy.

**Customer Segments from Uplift**:
- **Persuadables**: Positive uplift - will convert because of treatment (target these)
- **Sure Things**: Will convert regardless - marketing waste
- **Lost Causes**: Won't convert regardless - avoid marketing
- **Sleeping Dogs**: Negative uplift - marketing decreases conversion

**Meta-Learners in CausalML**:

| Learner | Approach | Best For | Limitation |
|---------|----------|----------|------------|
| **S-Learner** | Single model with treatment as feature | CATE=0, simple treatment effects | May ignore treatment if weak |
| **T-Learner** | Two separate models (treated/control) | Heterogeneous effects | Higher variance, unstable with small samples |
| **X-Learner** | Imputes ITE, then models residuals | Unbalanced treatment assignment | Requires good control model |
| **R-Learner** | Directly models treatment effect residual | High-dimensional confounders | Sensitive to propensity score estimation |

**Evidence on performance**:

```
Claim: In comparative benchmarks, X-learner consistently outperformed in large-sample settings while T-learner performed best at sample size 500 with heterogeneous effects [^9^]
Source: MDPI Engineering Proceedings
URL: https://www.mdpi.com/2297-8747/30/6/139
Date: 2025
Excerpt: "In Cases 4 and 5, both the causal forest and X-learner methods display strong estimation capabilities, with the X-learner consistently outperforming others in large-sample settings."
Confidence: High
```

### 2.2 DoWhy: Microsoft's Causal Inference Library

**Core Workflow** (four steps):

1. **Model**: Create causal graph encoding assumptions
2. **Identify**: Use backdoor/frontdoor criteria to identify estimand
3. **Estimate**: Apply statistical methods (can use EconML/CausalML)
4. **Refute**: Run robustness checks

**Supported Estimation Methods**:
- Propensity-based: stratification, matching, IPW
- Outcome modeling: linear regression, GLM
- Instrumental variables: 2SLS, Wald estimator
- Advanced: regression discontinuity, mediation analysis
- External: EconML (DML, CausalForest), CausalML (meta-learners)

**Refutation Tests**:
- **Placebo Treatment**: Replace real treatment with random, expect effect near zero
- **Random Common Cause**: Add random confounder, expect estimate stable
- **Data Subset**: Re-estimate on subsets, expect consistency
- **Bootstrap**: Re-sample and re-estimate

### 2.3 Causal Discovery (PC Algorithm, GES, Tigramite)

**PC Algorithm**:
- Constraint-based: uses conditional independence tests
- Requires: Causal Markov Condition + Faithfulness + No hidden confounders
- Limitation with 2-month data: needs n >> d (more samples than variables)
- Typical alpha: 0.05 for independence test threshold
- Tests available: Fisher-Z (default for continuous), Chi-squared, G-squared, KCI (kernel-based, more powerful but slower)

**Tigramite / PCMCI+** (for time series):
- Specifically designed for time series causal discovery
- Handles lagged relationships (demand today affects demand tomorrow)
- Tests: ParCorr (linear), GPDC (nonlinear), CMIknn (nonparametric)
- Can handle both lagged and contemporaneous links
- Assumes causal stationarity (stable causal structure over time)

**Sample Size Reality Check**:

```
Claim: For small datasets (L < 300), CDS algorithm is recommended; for time series, PCMCI achieves highest F1 with lowest runtime; MVGC is best for small sample sizes [^10^]
Source: Comprehensive Review of Causal Discovery Algorithms
URL: https://arxiv.org/html/2407.13054v2
Date: 2024
Excerpt: "CDS is more recommended in small size datasets...MVGC being more suitable for small sample sizes"
Confidence: High
```

### 2.4 Difference-in-Differences (DiD)

**Concept**: Compare demand changes before/after an event between a treated group and an untreated control group.

**Classic Example - Hotel Pricing**:

```
Claim: A DiD approach comparing demand before/after hotel price changes, using similar rooms without price changes as control, can estimate price elasticity from observational data [^11^]
Source: CESifo Working Paper
URL: https://www.econstor.eu/bitstream/10419/236669/1/cesifo1_wp9127.pdf
Date: 2022
Excerpt: "We restrict our analysis to 'accept-all' events...comparing demand and actual rate changes over those two days, using similar rooms in other hotels without an 'accept-all' event as the control group."
Confidence: High
```

**Implementation Formula**:
```
Treatment Effect = (Demand_treated_after - Demand_treated_before) - (Demand_control_after - Demand_control_before)
```

### 2.5 Synthetic Control

**Concept**: Create a weighted combination of control units that closely matches the pre-intervention trajectory of the treated unit, then project forward.

**Amazon's Graph Causal Forecasting (GCF)**:

```
Claim: GCF uses RGCN-dilated CNN network leveraging domain knowledge to automatically design a synthetic control during training, achieving 75.3% lower MAPE than baseline and 25% lower MSE than Google Causal Impact [^1^]
Source: AAAI 2025
URL: https://ojs.aaai.org/index.php/AAAI/article/view/35148
Date: 2025
Excerpt: "Our approach uses RGCN-dilated CNN based network, which leverages domain knowledge to automatically design a synthetic control during training...30% lower MSE against TGCN...25% lower MSE against Google Causal Impact model"
Confidence: High
```

**Traditional Synthetic Control**: Select weights w_j for control units to minimize pre-intervention distance to treated unit.

### 2.6 Causal Impact Analysis (Google's Approach)

**Concept**: Uses Bayesian Structural Time Series (BSTS) to predict what demand would have been without an intervention, comparing to actual observed demand.

**Python Implementation**: `pycausalimpact` package

**Critical Requirements**:
- **3:1 rule**: Pre-intervention period should be at least 3x the post-intervention period
- Control variables must NOT be affected by the intervention
- Minimum 30-50 pre-period observations
- For weekly data: 6-12 months of history recommended

**Applications to Demand**:
- Measure promotion incremental lift vs. cannibalization
- Quantify holiday effects on demand
- Evaluate impact of competitor entry/exit

```
Claim: 26% of promotions drain profits because they merely shift purchases from other time periods or cannibalize existing demand [^12^]
Source: Quantix AI - Causal Impact Analysis
URL: https://quantix-ai.eu/causal-impact-analysis-prove-which-marketing-actually-drives-sales/
Date: 2025
Excerpt: "CPG companies invest up to 20% of gross revenues on promotions, yet analysis by pricing optimization experts found that 26% of promotions drain profits past viability"
Confidence: Medium
```

### 2.7 Instrumental Variables (2SLS)

**Concept**: Use an external variable (instrument) that affects the treatment but not the outcome directly, to isolate causal variation.

**Classic Example - Fulton Fish Market**:

```
Claim: Stormy weather (wave height) was used as an instrument for fish supply price to estimate demand curve - weather affects supply but not demand directly [^13^]
Source: Instrumental Variables Course Materials
URL: https://rtgodwin.com/3040/iv.pdf
Date: Current
Excerpt: "Supply is affected by the weather: if the sea is rough it is harder to fish. Using the variables wave2 and wave3 as instruments for price, we can use variations in price that are due to changes in supply only, in order to estimate the slope of the demand curve."
Confidence: High
```

**Valid Instruments for Demand Prediction**:
- **Weather**: Temperature, rainfall, storms - affects demand but not influenced by business decisions
- **Holidays**: Exogenous timing, affects demand patterns
- **Competitor actions**: If truly independent of own demand
- **Policy changes**: Government regulations, tax changes

**Two-Stage Process**:
1. Stage 1: Treatment ~ Instrument + Controls (predict treatment from instrument)
2. Stage 2: Outcome ~ Predicted Treatment + Controls (use predicted values to estimate causal effect)

---

## 3. Implementation Details

### 3.1 CausalML Installation and Basic Usage

```bash
pip install causalml scikit-uplift
```

```python
# Uplift Random Forest for Demand Prediction
import numpy as np
import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import plot_gain, auuc_score
from sklearn.model_selection import train_test_split

# Features: customer/demand features
# treatment: binary (promotion applied or not)
# demand: binary (purchase or not, or above threshold)
X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
    df.drop(columns=['demand', 'treatment']), 
    df['demand'], 
    df['treatment'],
    test_size=0.3, 
    random_state=42
)

# Train Uplift Random Forest
uplift_rf = UpliftRandomForestClassifier(
    control_name='control',
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=200,
    evaluationFunction='KL'
)
uplift_rf.fit(X_train.values, treatment=treatment_train.values, y=y_train.values)

# Predict uplift
y_pred = uplift_rf.predict(X_test.values)

# Evaluate with AUUC
uplift_results = pd.DataFrame(y_pred, columns=uplift_rf.classes_[1:])
auuc_metrics = (uplift_results.assign(
    is_treated=(treatment_test.values != 'control').astype(int),
    conversion=y_test.values,
    uplift=uplift_results.max(axis=1)
))
score = auuc_score(auuc_metrics, outcome_col='conversion', treatment_col='is_treated')
```

### 3.2 EconML for Price Elasticity (Double ML)

```python
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV

# Double ML for price elasticity estimation
# y: demand (outcome)
# T: price (treatment)
# X: features (heterogeneity dimensions)
# W: controls (confounders)

# Linear DML - good for average treatment effect
dml = LinearDML(
    model_y=GradientBoostingRegressor(),  # Predict demand
    model_t=GradientBoostingRegressor(),  # Predict price
    model_final=LassoCV(),                # Treatment effect model
    cv=5
)
dml.fit(y, T, X=X, W=W)

# Get average treatment effect (price elasticity)
ate = dml.ate_
ate_ci = dml.ate_interval(alpha=0.05)

# Causal Forest DML - for heterogeneous effects
cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingRegressor(),
    cv=5,
    criterion='mse',
    n_estimators=1000,
    min_samples_leaf=10
)
cf_dml.fit(y, T, X=X, W=W)

# Get heterogeneous treatment effects
te = cf_dml.effect(X_test)
```

### 3.3 DoWhy Complete Workflow

```python
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import GradientBoostingRegressor
from econml.dml import DML

# Step 1: Model - Define causal graph
gml_graph = """
digraph {
    price -> demand;
    promotion -> demand;
    holiday -> demand;
    weather -> demand;
    seasonality -> demand;
    competitor_price -> demand;
    seasonality -> price;
    seasonality -> promotion;
}
"""

model = CausalModel(
    data=df,
    treatment='price',          # What we want to change
    outcome='demand',           # What we want to predict
    graph=gml_graph,
    common_causes=['seasonality', 'holiday', 'weather'],
    instruments=['competitor_price']  # IV if available
)

# Visualize the causal graph
model.view_model()

# Step 2: Identify - Find valid estimand
identified_estimand = model.identify_effect()
print(identified_estimand)

# Step 3: Estimate - Multiple methods for robustness
# Method 1: Linear regression
estimate_reg = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    confidence_intervals=True
)

# Method 2: Propensity score stratification
estimate_ipw = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_stratification"
)

# Method 3: Double Machine Learning via EconML
dml_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    method_params={
        "init_params": {
            'model_y': GradientBoostingRegressor(),
            'model_t': GradientBoostingRegressor(),
            'model_final': LassoCV()
        },
        "fit_params": {}
    }
)

# Step 4: Refute - Robustness checks
refute_placebo = model.refute_estimate(
    identified_estimand, estimate_reg,
    method_name="placebo_treatment_refuter",
    placebo_type="permute", 
    num_simulations=100
)

refute_random = model.refute_estimate(
    identified_estimand, estimate_reg,
    method_name="random_common_cause",
    num_simulations=100
)

refute_subset = model.refute_estimate(
    identified_estimand, estimate_reg,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    num_simulations=100
)

print(f"Placebo p-value: {refute_placebo.p_value}")
print(f"Random cause p-value: {refute_random.p_value}")
```

### 3.4 Google Causal Impact (pycausalimpact)

```python
from causalimpact import CausalImpact
import pandas as pd

# Data: demand series + control covariates
# y: demand for product with promotion (treatment)
# X: demand for similar products without promotion (controls)

data = pd.DataFrame({'y': treated_demand, 'X1': control_demand_1, 'X2': control_demand_2})

# Define pre/post periods (indices)
pre_period = [0, 59]     # 60 days before promotion
post_period = [60, 89]   # 30 days during promotion

# Run causal impact
impact = CausalImpact(data, pre_period, post_period)

# Summary
print(impact.summary())
print(impact.summary('report'))

# Plot
impact.plot()

# Access results
print(f"Absolute effect: {impact.summary_data['abs_effect']}")
print(f"Relative effect: {impact.summary_data['rel_effect']}%")
print(f"P-value: {impact.p_value}")
```

### 3.5 Tigramite for Time Series Causal Discovery

```python
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

# Prepare data (time series matrix: T x N)
data_pp = pp.DataFrame(df.values, var_names=df.columns)

# Choose independence test (ParCorr for linear)
parcorr = ParCorr()

# Run PCMCI+
pcmci = PCMCI(dataframe=data_pp, cond_ind_test=parcorr)

# Run with tau_max (maximum time lag)
results = pcmci.run_pcmciplus(tau_min=0, tau_max=7, pc_alpha=0.05)

# Extract causal graph
graph = results['graph']  # graph[i,j,tau] = '->', '<-', 'o-o', or ''
val_matrix = results['val_matrix']  # test statistic values
p_matrix = results['p_matrix']  # p-values

# Plot results
from tigramite import plotting as tpl
link_matrix = pcmci.return_significant_links(
    pq_matrix=results['p_matrix'], 
    val_matrix=results['val_matrix'], 
    alpha_level=0.05
)['link_matrix']

tpl.plot_graph(
    val_matrix=val_matrix,
    link_matrix=link_matrix,
    var_names=df.columns,
    link_colorbar_label='cross-MCI'
)
```

### 3.6 Instrumental Variables (2SLS)

```python
import linearmodels
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

# First stage: Predict treatment (price) from instrument (weather)
first_stage = sm.OLS(df['price'], 
                     sm.add_constant(df[['temperature', 'controls']])).fit()
df['price_hat'] = first_stage.predict(sm.add_constant(df[['temperature', 'controls']]))

# Check instrument strength (F-stat should be > 10)
print(f"First stage F-stat: {first_stage.fvalue}")

# Second stage: Predict demand from predicted price
second_stage = sm.OLS(df['demand'], 
                      sm.add_constant(df[['price_hat', 'controls']])).fit()
print(second_stage.summary())

# Alternatively using linearmodels
iv_model = IV2SLS(
    dependent=df['demand'],
    exog=sm.add_constant(df['controls']),
    endog=df['price'],
    instruments=df[['temperature', 'rainfall']]
).fit()
print(iv_model.summary)
```

### 3.7 Difference-in-Differences

```python
import statsmodels.formula.api as smf

# Create treatment and post indicators
df['post'] = (df['date'] >= event_date).astype(int)
df['treatment'] = df['is_treated']
df['did'] = df['post'] * df['treatment']

# DiD regression
model = smf.ols('demand ~ treatment + post + did + C(store_id) + C(date)', 
                data=df).fit()
print(model.summary())

# The coefficient on 'did' is the treatment effect
did_effect = model.params['did']
print(f"Difference-in-Differences Effect: {did_effect}")
```

---

## 4. What Works

### 4.1 Best Practices from Evidence

1. **Use DoWhy's refutation tests**: Always run placebo, random common cause, and subset tests to validate causal estimates. Passing these tests dramatically increases confidence.

2. **Apply Double ML for price elasticity**: EconML's DML methods provide the most robust price elasticity estimates because they handle high-dimensional confounders and avoid overfitting.

3. **Use Causal Impact for promotion/holiday effects**: When you have sufficient pre-period data (3:1 ratio), Causal Impact provides clean estimates of intervention effects.

4. **Combine multiple estimation methods**: Run regression adjustment, IPW, and doubly robust estimators. If all converge, confidence is higher.

5. **Weather as instrument**: When you need to isolate supply-side or demand-side effects, weather provides a plausibly exogenous source of variation.

6. **Heterogeneous effects with Causal Forest**: When treatment effects vary by customer/product segment, CausalForestDML from EconML captures this heterogeneity.

7. **Uplift modeling for targeting**: CausalML's uplift trees identify which customers respond to promotions, improving marketing ROI by 20-40%.

### 4.2 Proven Applications in Demand Forecasting

| Application | Method | Result | Source |
|-------------|--------|--------|--------|
| Amazon product demand with suppressed listings | Graph Causal Forecasting | 75.3% MAPE reduction | AAAI 2025 [^1^] |
| Ride-sharing surge pricing effects | DiD + Causal Forest | Cherry-picking vs. competition effects identified | UCL/Management Science [^14^] |
| Promotion cannibalization | Causal Impact per product pair | Directed graph of cannibalization | IEEE Access [^7^] |
| Price elasticity by product group | EconML DML | Elasticity by segment | EIA University [^15^] |
| Hotel pricing demand estimation | DiD | Quasi-experimental price elasticity | CESifo [^11^] |
| Grocery promotion targeting | Uplift Modeling | 20-40% marketing efficiency gain | Industry reports [^16^] |
| Ride-sharing pricing optimization | Generalized Causal Forest | 15-25% order improvement | KDD 2022 [^6^] |

### 4.3 Integration Pipeline

```
Raw Data -> Feature Engineering -> Causal Discovery (Tigramite) -> Causal Graph (DoWhy) ->
Effect Estimation (EconML/CausalML) -> Refutation Tests (DoWhy) ->
Causal-Enhanced Forecast -> Business Decision
```

---

## 5. What Doesn't Work

### 5.1 Common Pitfalls

1. **Causal discovery with < 100 samples per variable**: PC algorithm and most causal discovery methods require n >> d. With 2 months of daily data (60 samples) and 20+ features, causal discovery will produce unreliable graphs.

2. **Using Causal Impact with insufficient pre-period data**: Having equal pre/post periods or more post than pre data starves the model of pattern-learning capability.

3. **Contaminated control variables in Causal Impact**: Using variables affected by the intervention (e.g., website traffic as control for a marketing campaign) produces biased underestimates.

4. **Weak instruments**: Instrumental variables with first-stage F-stat < 10 produce biased estimates. Always check instrument strength.

5. **Ignoring confounding**: Simple correlation between price and demand is confounded by seasonality, competition, and product lifecycle. Without controlling for these, estimates are meaningless.

6. **Using uplift models without randomization**: Uplift modeling assumes randomized treatment assignment. With observational data, use propensity score methods or DiD instead.

7. **Static causal graphs for time-varying relationships**: Standard causal discovery assumes a static DAG. For demand data with changing relationships over time, use time-series-specific methods (Tigramite) or regime-specific analysis.

### 5.2 What to Avoid with Limited Data

| Method | 2-Month Daily Data | Recommendation |
|--------|-------------------|----------------|
| PC Algorithm (causal discovery) | Unreliable | Use only with strong domain constraints |
| Causal Impact | Marginal (if 6+ weeks pre) | Needs minimum 3:1 pre/post ratio |
| Double ML | Viable with regularization | Use Lasso/Ridge in base learners |
| DiD | Good | Needs at least some pre/post data |
| 2SLS/IV | Viable | Depends on instrument quality |
| Uplift Modeling | Only if > 1000 observations | Needs sufficient sample for train/test |

### 5.3 Failed Approaches to Watch

```
Claim: Matching methods (propensity score matching) often fail in small samples - in simulations with N=100, matching had the highest failure rates and widest confidence intervals [^17^]
Source: PMC - Causal inference methods for small non-randomized studies
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7834813/
Date: 2020
Excerpt: "Matching had 1366 failures out of simulations...for small sample sizes, the logistic regression model often failed"
Confidence: High
```

---

## 6. Competition Applications

### 6.1 How Causal Inference Applies to Forecasting Competitions

**M5 Competition Context**:

```
Claim: M5 was the first Makridakis competition to provide exogenous variables and hierarchical retail data, enabling causal modeling approaches [^18^]
Source: International Journal of Forecasting (M5 Competition paper)
URL: https://www.sciencedirect.com/science/article/pii/S0169207021001187
Date: 2022
Excerpt: "In contrast to the previous M competitions, teams were provided with exogenous/explanatory variables, besides time series data, that could be used to improve forecasting performance"
Confidence: High
```

### 6.2 Recommended Competition Strategy

For a 2-month demand forecasting problem, the recommended causal-enhanced approach:

1. **Use Causal Impact for known interventions**: If promotions, holidays, or events occurred during the data period, quantify their effect and remove from trend

2. **Apply DiD if control groups exist**: If some stores/products didn't receive treatment, use as natural control group

3. **Use IV approach with weather data**: If weather data is available, use temperature/holiday as instruments to isolate exogenous demand variation

4. **Estimate price elasticity with Double ML**: If price variation exists, use EconML DML to estimate causal price effect rather than simple correlation

5. **Incorporate causal effects as features**: Rather than just including promotion as a binary feature, include estimated causal effect magnitude

6. **Build uplift-aware forecasts**: Predict not just "what will demand be" but "what will demand be under each treatment scenario"

### 6.3 Academic Impact

Using causal inference in a forecasting competition signals:
- Understanding of structural vs. reduced-form models
- Ability to think counterfactually
- Knowledge of cutting-edge methods (DoWhy, EconML)
- Rigorous approach to feature importance (causal vs. correlational)

---

## 7. Recommended Approach for Our Problem

### 7.1 Tiered Implementation Strategy

**Tier 1: Causal Impact Analysis (Easiest, Most Impressive)**
- Use pycausalimpact to measure effects of known interventions (holidays, promotions)
- Provides counterfactual forecasts that account for what demand "would have been"
- Requires: pre-period data, control covariates

**Tier 2: DoWhy + EconML Integration**
- Build causal graph of demand drivers
- Use Double ML for price/promotion effect estimation
- Run refutation tests for credibility
- Integrate estimated causal effects as features in forecasting model

**Tier 3: Full Causal Pipeline**
- Use Tigramite for time-series causal discovery
- Build DAG from discovered relationships
- Estimate heterogeneous effects with Causal Forest
- Generate counterfactual scenarios ("what if we increase price by 10%?")

### 7.2 Quick-Start Code for Competition

```python
# Recommended minimal implementation for 2-month demand data
import pandas as pd
from causalimpact import CausalImpact
import dowhy
from dowhy import CausalModel
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV

# 1. Identify intervention periods (holidays, promotions)
holiday_dates = ['2023-12-25', '2024-01-01']  # example

# 2. Causal Impact for each holiday
for holiday in holiday_dates:
    pre_start, pre_end = get_pre_period(holiday, days=14)
    post_start, post_end = get_post_period(holiday, days=7)
    
    impact = CausalImpact(
        data[['demand', 'control_1', 'control_2']],
        [pre_start, pre_end],
        [post_start, post_end]
    )
    holiday_effects[holiday] = impact.summary_data['rel_effect']

# 3. Build causal graph
causal_graph = """
digraph {
    price -> demand;
    promotion -> demand;
    day_of_week -> demand;
    month -> demand;
    holiday -> demand;
    temperature -> demand;
}
"""

model = CausalModel(
    data=df,
    treatment='promotion',
    outcome='demand',
    graph=causal_graph
)

# 4. Estimate causal effects
dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=50),  # Small for limited data
    model_t=GradientBoostingRegressor(n_estimators=50),
    model_final=LassoCV(),
    cv=3
)
dml.fit(df['demand'], df['promotion'], X=df[['price']], W=df[['day_of_week', 'temperature']])

# 5. Use causal effects as features
df['causal_promotion_effect'] = dml.effect(df[['price']])
df['causal_price_elasticity'] = dml.coef_

# 6. Feed causal-enhanced features into forecasting model
features = df[['promotion', 'causal_promotion_effect', 'price', 'day_of_week']]
```

### 7.3 Key Advantages of This Approach

1. **Intellectual sophistication**: Goes far beyond standard ML forecasting
2. **Business relevance**: Answers "what if" questions, not just "what will"
3. **Robustness**: DoWhy's refutation tests validate findings
4. **Interpretability**: Causal graphs make assumptions explicit
5. **Competition edge**: Most teams won't use causal inference - this differentiates

---

## 8. Sources

[^1^] Basu, S., Kumar, M., & Kaveri, S. (2025). GCF: Estimating Unobserved Demand Using Graph Causal Forecasting. Proceedings of the AAAI Conference on Artificial Intelligence, 39(28), 28836-28842. https://ojs.aaai.org/index.php/AAAI/article/view/35148

[^2^] Sharma, A. & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference. Microsoft Research. https://arxiv.org/pdf/2011.04216

[^3^] Comprehensive Review and Empirical Evaluation of Causal Discovery Algorithms for Numerical Data (2024). https://arxiv.org/html/2407.13054v2

[^4^] MCP Analytics (2025). Causal Impact Explained (with Examples). https://mcpanalytics.ai/articles/causal-impact-practical-guide-for-data-driven-decisions

[^5^] Auffhammer, M., Hsiang, S.M., Schlenker, W., & Sobel, A. (2013). Using Weather Data and Climate Model Output in Economic Analyses of Climate Change. Review of Environmental Economics and Policy, 7(2), 181-198. https://www.nber.org/system/files/working_papers/w19087/w19087.pdf

[^6^] Wan, S., Zheng, C., Sun, Z. et al. (2022). GCF: Generalized Causal Forest for Heterogeneous Treatment Effect Estimation in Online Marketplace. KDD 2022 Workshop. https://arxiv.org/abs/2203.10975

[^7^] Causal Quantification of Cannibalization During Promotional Sales in Grocery Retail (2021). IEEE Access, 9, 34078-34089. https://doi.org/10.1109/ACCESS.2021.3062222

[^8^] EconML Documentation: Orthogonal/Double Machine Learning. https://econml.azurewebsites.net/spec/estimation/dml.html

[^9^] Comparing Meta-Learners for Estimating Heterogeneous Treatment Effects (2025). MDPI Engineering Proceedings, 30(6), 139. https://www.mdpi.com/2297-8747/30/6/139

[^10^] Comprehensive Review and Empirical Evaluation of Causal Discovery Algorithms (2024). https://arxiv.org/html/2407.13054v2

[^11^] Demand Estimation Using Managerial Responses to Automated Price Recommendations (2022). CESifo Working Paper No. 9127. https://www.econstor.eu/bitstream/10419/236669/1/cesifo1_wp9127.pdf

[^12^] Quantix AI (2025). Causal Impact Analysis: Prove Which Marketing Actually Drives Sales. https://quantix-ai.eu/causal-impact-analysis-prove-which-marketing-actually-drives-sales/

[^13^] Instrumental Variables / Two-stage least-squares course materials. https://rtgodwin.com/3040/iv.pdf

[^14^] Miao, W., Deng, Y., Wang, W., Liu, Y., & Tang, C.S. (2022). The effects of surge pricing on driver behavior in the ride-sharing market: Evidence from a quasi-experiment. https://discovery.ucl.ac.uk/id/eprint/10156986/

[^15^] EIA University. Price Elasticity of Demand Estimation using EconML. https://repository.eia.edu.co/bitstreams/cfda4ffc-1242-4b41-95fc-ee11bf471470/download

[^16^] Uplift Modeling: Maximizing Marketing ROI Through Causal Inference. https://cooptml.com/blog/uplift-modeling

[^17^] Causal inference methods for small non-randomized studies (2020). PMC, 7834813. https://pmc.ncbi.nlm.nih.gov/articles/PMC7834813/

[^18^] Makridakis, S. et al. (2022). The M5 competition: Background, organization, and implementation. International Journal of Forecasting. https://www.sciencedirect.com/science/article/pii/S0169207021001187

[^19^] Causal Inference: Tigramite Documentation. https://github.com/jakobrunge/tigramite

[^20^] Causal Impact Python Package. https://github.com/pombredanne/causalimpact-1

[^21^] CausalML Documentation. https://causalml.readthedocs.io/en/stable/

[^22^] DoWhy Documentation. https://www.pywhy.org/dowhy/

[^23^] Runge, J. et al. (2019). Inferring causation from time series in Earth system sciences. Nature Communications, 10(1), 2553. https://www.nature.com/articles/s41467-019-10105-3

[^24^] Gutierrez, P. & Gerardy, J.Y. (2017). Causal Inference and Uplift Modeling: A review of the literature. Proceedings of Machine Learning Research, 67. https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf

[^25^] Causal Inference and Model Explainability Tools for Retail (2025). https://arxiv.org/html/2512.12605v1

[^26^] Optimizing Enterprise Decision-Making through Causal Machine Learning (2025). Zenodo. https://zenodo.org/records/15386602

[^27^] Sundell, M. & Abeln, M. (2023). Explainable Demand Forecasting using Causalities and Machine Learning Models. Lund University. https://lup.lub.lu.se/student-papers/record/9127837/file/9127846.pdf

[^28^] SoCalGas Demand Response Load Impact Evaluation (2018). https://www.calmac.org/publications/

[^29^] A Clustering Method for Product Cannibalization Detection Using Price Effect (2025). Electronics, 14(15), 3120. https://www.mdpi.com/2079-9292/14/15/3120

[^30^] Brodersen, K.H. et al. Inferring causal impact using Bayesian structural time-series models. Google. https://searchengineland.com/causal-impact-studies-446862

---

## Appendix: Quick Reference Card

### Library Installation
```bash
pip install causalml econml dowhy pycausalimpact tigramite linearmodels
```

### When to Use Each Method

| Question | Method | Library |
|----------|--------|---------|
| "Who responds to promotions?" | Uplift Modeling | CausalML |
| "What is the causal graph?" | Causal Discovery | Tigramite, PC (causal-learn) |
| "Did holiday X affect demand?" | Causal Impact | pycausalimpact |
| "What is price elasticity?" | Double ML | EconML |
| "What if we changed price?" | Counterfactual | DoWhy + EconML |
| "Did promotion help or hurt?" | DiD | statsmodels |
| "How to isolate price effect?" | Instrumental Variables | linearmodels |
| "Which features drive demand?" | Causal Discovery | Tigramite, CausalNex |

### Key Parameters for 2-Month Data

| Method | Minimum Samples | Key Parameter |
|--------|----------------|---------------|
| Causal Impact | 30 pre, 10 post | pre_period >= 3x post_period |
| DML | 100+ observations | cv=3 (small), regularization in base learners |
| DiD | Some pre + post | Include fixed effects |
| Uplift Forest | 500+ observations | min_samples_leaf=50 (small data) |
| PC Algorithm | n >> d | alpha=0.10 (more lenient), use background knowledge |
| 2SLS | 100+ observations | Check F-stat > 10 |
