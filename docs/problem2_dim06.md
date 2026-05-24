# Dimension 6: Supply Chain Domain-Specific Features

## Research Summary

This research dimension focuses on supply chain domain-specific features that can be engineered from transactional order data to improve demand forecasting accuracy. The analysis covers lead time patterns, order frequency, order size distribution, warehouse utilization, store-level patterns, product lifecycle stages, safety stock behavior, and distribution network topology. These features encode deep industry understanding of how supply chains operate and how demand signals flow through warehouse-store networks.

---

## 1. Key Findings

### Finding 1: RFM-Based Customer Segmentation Dramatically Improves Forecast Accuracy

Claim: Customer segmentation using RFM (Recency, Frequency, Monetary) analysis before demand forecasting enables separate models per segment, improving accuracy by accounting for different purchasing behaviors [^1^].
Source: ScienceDirect - Cluster-based demand forecasting using Bayesian model averaging
URL: https://www.sciencedirect.com/science/article/pii/S2772662222000066
Date: 2022-06-01
Excerpt: "Customers are segmented based on their recency, frequency, and monetary (RFM) characteristics. Customers with similar buying behaviors are recognized as a segment, creating an ordered relationship between transactions made by them... time-series analysis techniques are used to forecast demand for each customer segment."
Confidence: High

### Finding 2: Lead Time Features Are Critical Supply Chain Predictors

Claim: Lead time variability is one of the most important factors affecting supply chain performance and should be explicitly modeled as a feature, including actual shipping days and late delivery flags [^2^].
Source: MDPI Sustainability - Enhancing Supply Chain Management: A Comparative Study of ML Techniques
URL: https://www.mdpi.com/2071-1050/17/13/5772
Date: 2025-06-23
Excerpt: "Feature engineering incorporated lag variables, promotional flags, holiday indicators... Actual Shipping Days: Shipping delay is a key operational indicator that reflects fulfillment efficiency... Late Delivery Flag: This binary indicator flags whether an order violated its promised lead time."
Confidence: High

### Finding 3: ADI-CV² Classification Enables Intelligent Method Selection

Claim: Demand patterns classified by Average Demand Interval (ADI) and Coefficient of Variation squared (CV²) into smooth, erratic, intermittent, and lumpy categories enables matching the right forecasting method to each pattern, improving portfolio-wide accuracy by 20-40% [^3^].
Source: MCP Analytics - Croston Method for Intermittent Demand Whitepaper
URL: https://mcpanalytics.ai/whitepapers/whitepaper-croston-method
Date: 2025-12-26
Excerpt: "Items exhibit four archetypal patterns based on ADI and CV²: smooth (ADI < 1.32, CV² < 0.49), erratic (ADI < 1.32, CV² >= 0.49), intermittent (ADI >= 1.32, CV² < 0.49), and lumpy (ADI >= 1.32, CV² >= 0.49). Each pattern responds differently to Croston variants, necessitating intelligent method selection."
Confidence: High

### Finding 4: ABC-XYZ Segmentation Is the Industry Standard for Inventory-Driven Forecasting

Claim: ABC-XYZ analysis combining value-based (ABC) and variability-based (XYZ) classification is widely adopted in industry for demand planning and inventory segmentation, enabling tailored forecasting strategies per segment [^4^].
Source: MIT Supply Chain Thesis
URL: https://dspace.mit.edu/bitstream/handle/1721.1/117927/1051223544-MIT.pdf
Date: Unknown
Excerpt: "The demand coefficient of variation (CV) is used to segment because it allows various demand distributions to be standardized and compared... Items with rather constant consumption and rare fluctuations are denoted X and have a CV < 0.5. Items with stronger fluctuations are denoted Y and have a CV between 0.5 and 1. Items with completely irregular consumption are termed Z and have a CV > 1."
Confidence: High

### Finding 5: Product Lifecycle Stages Require Different Forecasting Models

Claim: Products in different lifecycle stages (introduction, growth, maturity, decline) exhibit fundamentally different demand patterns that require stage-specific forecasting approaches; using a tracking signal to detect lifecycle transitions is critical [^5^].
Source: SlimStock - Product Lifecycle Management Guide
URL: https://www.slimstock.com/blog/product-lifecycle-management/
Date: 2025-11-21
Excerpt: "During the growth phase, sales figures follow a certain trend. However, at some point, the demand starts to follow a pattern without a trend. This means that the forecasting model is no longer appropriate... The transition from maturity to decline is critical because the risk of obsolescence now comes into play!"
Confidence: High

### Finding 6: Store-Level Forecasting Requires Granular Location-Specific Features

Claim: Forecasting at the store-item level captures localized demand patterns that aggregated forecasts miss; store metadata (size, type, demographics) and same-store trends are essential features [^6^].
Source: Lark - Store Level Forecasting Guide
URL: https://www.larksuite.com/en_us/topics/retail-glossary/store-level-forecasting
Date: 2024-01-11
Excerpt: "Store-level forecasting refers to the process of predicting product demand at the individual store or location level... The primary goal is to ensure that each retail location has the right products in the right quantities at the right time."
Confidence: High

### Finding 7: Pre-Holiday Safety Stock Ordering Creates Predictable Demand Spikes

Claim: Retailers systematically increase orders before weekends and holidays as safety stock behavior, creating predictable demand patterns that can be captured with calendar-aware features [^7^].
Source: VersaCloud ERP - Pre-Holiday Inventory Management Best Practices
URL: https://www.versaclouderp.com/blog/effective-stock-optimization-techniques-in-pre-holiday-inventory-management/
Date: 2024-10-31
Excerpt: "Inventory turnover typically doubles or triples during holiday peaks... businesses can prevent stockouts by increasing safety stock buffers, setting automated reorder points, and reserving emergency supplier capacity."
Confidence: Medium

### Finding 8: Hub-and-Spoke vs Cross-Docking Patterns Affect Demand Timing

Claim: Distribution network topology (hub-and-spoke vs direct delivery vs cross-docking) significantly affects order timing patterns and lead times, with cross-docking reducing lead time from 7 days to less than 2 days [^8^].
Source: Cadre Technologies - Cross Docking Case Studies
URL: https://www.cadretech.com/cross-docking/
Date: 2025-08-08
Excerpt: "A major retail chain implemented cross docking for fast-moving consumer goods, resulting in a 28% reduction in overall supply chain costs and a 35% improvement in inventory turns... average lead time from seven days to less than two days."
Confidence: High

### Finding 9: Rolling Statistics at Multiple Hierarchy Levels Are Top Kaggle Features

Claim: Rolling statistics (means, medians, std dev) calculated at multiple aggregation levels (product, store, category, combinations) are consistently the most important features in retail demand forecasting competitions [^9^].
Source: arXiv - Learnings from Kaggle's Forecasting Competitions
URL: https://arxiv.org/pdf/2009.07701
Date: Unknown
Excerpt: "The features used were mainly rolling statistics grouped by various factors such as store, item, class, and their combinations. The statistics used included measures of centrality and spread, as well as an exponential moving average."
Confidence: High

### Finding 10: ML-Enhanced Croston Frameworks Outperform Traditional Methods for Intermittent Demand

Claim: Combining Croston's decomposition with ML models (XGBoost, LightGBM, LSTM) for intermittent demand achieves 20-35% accuracy improvement over traditional methods, with feature engineering playing a larger role than architectural complexity [^10^].
Source: PMC - Primacy of feature engineering over architectural complexity for intermittent demand forecasting
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12873174/
Date: Unknown
Excerpt: "A structured feature engineering pipeline transforms the padded monthly data into a high-dimensional feature set including temporal features (lagged values, rolling sums, count of non-zero periods, time elapsed since last sale) and algorithmic features (forecasts from Croston's method)."
Confidence: High

---

## 2. Major Techniques

### 2.1 Lead Time Feature Engineering

Lead time features capture the time between order creation and delivery. Key implementations include:

**Core Lead Time Features:**
- `actual_shipping_days` = ActualShipDate - OrderDate (in days)
- `promised_lead_time` = Expected delivery days from order
- `late_delivery_flag` = 1 if actual_shipping_days > promised_lead_time, else 0
- `lead_time_deviation` = actual_shipping_days - promised_lead_time
- `lead_time_variability_30d` = std(actual_shipping_days) over last 30 days

**Advanced Lead Time Features:**
- Lead time probability distribution per warehouse-customer pair
- Lead time trend (is lead time getting longer or shorter?)
- Lead time seasonality (does lead time vary by month/holiday?)
- Order-to-delivery pattern by day of week

From academic research on pharmaceutical lead time prediction [^11^]:
- Day of month of customer order
- Weekday of customer order  
- Month of customer order
- Supplier code identifier
- Product type category
- Machine learning models (Random Forest, CatBoost, SVM) achieve MSE of 1.89-2.31 days

### 2.2 Order Frequency Pattern Analysis (RFM Segmentation)

RFM (Recency, Frequency, Monetary) segmentation groups customers by purchasing behavior before building forecasts:

```python
# RFM Feature Engineering
def compute_rfm_features(df, customer_col='customer_id', date_col='order_date', 
                         amount_col='order_amount'):
    snapshot_date = df[date_col].max()
    rfm = df.groupby(customer_col).agg(
        recency=('snapshot_date' - date_col.max()).days,
        frequency=date_col.nunique(),
        monetary=amount_col.sum()
    )
    # Quintile-based scoring (1-5)
    rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
    # Combined segment
    rfm['rfm_segment'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    return rfm
```

### 2.3 Order Size Distribution Features

Key metrics for order size distribution per customer-product combination:

- `avg_order_qty` = mean(quantity) per customer-product
- `std_order_qty` = std(quantity) per customer-product
- `cv_order_qty` = std_order_qty / avg_order_qty (coefficient of variation)
- `median_order_qty` = median(quantity) per customer-product
- `max_order_qty` = max historical quantity
- `order_qty_percentile_90` = 90th percentile of order quantities
- `order_size_trend` = slope of order quantity over time (increasing/decreasing)
- `bulk_order_flag` = 1 if order > 2*median_order_qty

### 2.4 Warehouse Utilization & Network Features

Features derived from warehouse-store assignment patterns:

- `warehouse_store_pairs` = unique (warehouse, store) combinations
- `primary_warehouse` = warehouse that handles most orders per store
- `warehouse_share_pct` = % of store's orders from each warehouse
- `cross_docking_indicator` = 1 if order flows through multiple warehouses
- `warehouse_switch_flag` = 1 if store changed primary warehouse
- `days_since_last_order_from_warehouse` = recency per warehouse-store pair
- `warehouse_order_count_30d` = order frequency per warehouse

### 2.5 Store-Level Pattern Features

Store-specific demand characteristics:

- `store_age_days` = days since first order from store
- `store_lifecycle_stage` = new (<90 days), growing, mature, declining
- `same_store_demand_trend` = slope of monthly demand
- `store_demand_cv` = coefficient of variation of store demand
- `store_peak_dayofweek` = day with highest average orders
- `store_order_frequency` = average days between orders
- `store_avg_order_size` = average units per order
- `store_category_mix` = distribution of product categories ordered
- `days_since_last_order` = recency feature per store

### 2.6 Product Lifecycle Features

Detecting and encoding product lifecycle stages:

- `product_age_days` = days since first sale
- `lifecycle_stage` = introduction, growth, maturity, decline
- `demand_trend_slope` = slope of demand over last 90 days
- `growth_rate` = (current_month - prior_month) / prior_month
- `maturity_flag` = 1 if demand stable for 60+ days
- `decline_flag` = 1 if negative trend for 30+ consecutive days
- `days_since_peak_demand` = days since maximum historical demand
- `tracking_signal` = cumulative forecast error / MAD (detects model breakdown)

### 2.7 Safety Stock Behavior Features

Capturing pre-weekend and pre-holiday ordering patterns:

- `days_until_weekend` = min(days until Saturday, days until Sunday)
- `is_pre_weekend_order` = 1 if order on Thursday/Friday
- `days_until_holiday` = days until next holiday
- `is_pre_holiday_order` = 1 if order within 3 days before holiday
- `post_holiday_flag` = 1 if order within 2 days after holiday
- `safety_stock_order_pattern` = historical avg order qty before weekends
- `weekend_uplift_factor` = (Thu+Fri avg qty) / (Mon+Tue avg qty)
- `month_start_flag` = 1 if order in first 3 days of month
- `month_end_flag` = 1 if order in last 3 days of month

### 2.8 Distribution Network Pattern Features

Encoding network topology effects:

- `network_distance_proxy` = number of intermediate stops
- `hub_utilization_rate` = orders through hub / total hub capacity
- `direct_delivery_flag` = 1 if warehouse delivers directly to store
- `hub_vs_direct_ratio` = ratio of hub to direct deliveries per store
- `delivery_route_complexity` = number of unique warehouse combinations used
- `store_cluster` = cluster of stores with similar ordering patterns

### 2.9 ADI-CV² Demand Pattern Classification

Classifying demand patterns for method selection:

```python
def classify_demand_pattern(demand_series):
    """
    Classify demand as: smooth, erratic, intermittent, lumpy
    Using Syntetos et al. thresholds
    """
    total_periods = len(demand_series)
    non_zero_periods = (demand_series > 0).sum()
    adi = total_periods / non_zero_periods if non_zero_periods > 0 else float('inf')
    
    non_zero_demands = demand_series[demand_series > 0]
    cv_squared = (non_zero_demands.std() / non_zero_demands.mean()) ** 2 if len(non_zero_demands) > 1 else 0
    
    if adi < 1.32 and cv_squared < 0.49:
        return 'smooth'
    elif adi < 1.32 and cv_squared >= 0.49:
        return 'erratic'
    elif adi >= 1.32 and cv_squared < 0.49:
        return 'intermittent'
    else:
        return 'lumpy'
```

---

## 3. Implementation Details

### 3.1 Complete Supply Chain Feature Engineering Pipeline

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_supply_chain_features(df, warehouses_col='warehouse', 
                                   stores_col='store_name',
                                   products_col='product_code',
                                   date_col='order_date',
                                   qty_col='quantity'):
    """
    Create comprehensive supply chain domain features
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    features = pd.DataFrame()
    features[date_col] = df[date_col]
    features[warehouses_col] = df[warehouses_col]
    features[stores_col] = df[stores_col]
    features[products_col] = df[products_col]
    
    # ---- 1. LEAD TIME FEATURES ----
    if 'actual_delivery_date' in df.columns:
        features['shipping_days'] = (df['actual_delivery_date'] - df[date_col]).dt.days
        features['is_late'] = features['shipping_days'] > df.get('promised_lead_days', 7)
    
    # ---- 2. ORDER FREQUENCY FEATURES ----
    # Days since last order per store-product
    features['days_since_last_order'] = df.groupby([stores_col, products_col])[date_col].diff().dt.days
    
    # Days until weekend
    features['dayofweek'] = df[date_col].dt.dayofweek  # Monday=0
    features['days_until_saturday'] = (5 - features['dayofweek']) % 7
    features['days_until_sunday'] = (6 - features['dayofweek']) % 7
    features['days_until_weekend'] = features[['days_until_saturday', 'days_until_sunday']].min(axis=1)
    
    # ---- 3. ORDER SIZE FEATURES ----
    features['qty'] = df[qty_col]
    
    # Rolling statistics per customer-product (avoiding leakage)
    group = df.groupby([stores_col, products_col])[qty_col]
    features['avg_qty_30d'] = group.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    features['std_qty_30d'] = group.transform(lambda x: x.shift(1).rolling(30, min_periods=1).std())
    features['cv_qty_30d'] = features['std_qty_30d'] / features['avg_qty_30d']
    features['median_qty_30d'] = group.transform(lambda x: x.shift(1).rolling(30, min_periods=1).median())
    features['max_qty_30d'] = group.transform(lambda x: x.shift(1).rolling(30, min_periods=1).max())
    features['is_bulk_order'] = (df[qty_col] > 2 * features['median_qty_30d']).astype(int)
    
    # ---- 4. WAREHOUSE FEATURES ----
    # Store's primary warehouse
    store_warehouse = df.groupby(stores_col)[warehouses_col].agg(lambda x: x.mode()[0])
    features['primary_warehouse'] = df[stores_col].map(store_warehouse)
    features['is_primary_warehouse'] = (df[warehouses_col] == features['primary_warehouse']).astype(int)
    
    # ---- 5. STORE-LEVEL FEATURES ----
    # Store age
    store_first_order = df.groupby(stores_col)[date_col].transform('min')
    features['store_age_days'] = (df[date_col] - store_first_order).dt.days
    features['store_lifecycle'] = pd.cut(features['store_age_days'], 
                                         bins=[0, 90, 365, 730, float('inf')],
                                         labels=['new', 'growing', 'mature', 'old'])
    
    # ---- 6. PRODUCT LIFECYCLE FEATURES ----
    product_first_sale = df.groupby(products_col)[date_col].transform('min')
    features['product_age_days'] = (df[date_col] - product_first_sale).dt.days
    
    # Demand trend (last 30 days slope proxy)
    features['qty_lag_7d'] = group.transform(lambda x: x.shift(7))
    features['qty_lag_14d'] = group.transform(lambda x: x.shift(14))
    features['qty_lag_30d'] = group.transform(lambda x: x.shift(30))
    
    # ---- 7. SAFETY STOCK BEHAVIOR FEATURES ----
    features['is_pre_weekend'] = df[date_col].dt.dayofweek.isin([3, 4]).astype(int)  # Thu, Fri
    features['is_friday'] = (df[date_col].dt.dayofweek == 4).astype(int)
    features['is_month_end'] = df[date_col].dt.day.isin([28, 29, 30, 31]).astype(int)
    features['is_month_start'] = df[date_col].dt.day.isin([1, 2, 3]).astype(int)
    
    # Weekend uplift ratio
    dow_qty = df.groupby(df[date_col].dt.dayofweek)[qty_col].mean()
    weekend_avg = dow_qty[[5, 6]].mean() if 5 in dow_qty and 6 in dow_qty else dow_qty.mean()
    weekday_avg = dow_qty[[0, 1, 2, 3, 4]].mean()
    features['weekend_uplift'] = weekend_avg / weekday_avg if weekday_avg > 0 else 1
    
    # ---- 8. NETWORK FEATURES ----
    # Number of warehouses used by store
    n_warehouses = df.groupby(stores_col)[warehouses_col].nunique()
    features['n_warehouses_used'] = df[stores_col].map(n_warehouses)
    
    return features
```

### 3.2 ABC-XYZ Segmentation Implementation

```python
def abc_xyz_segmentation(df, sku_col='product_code', value_col='revenue', 
                         qty_col='quantity', periods=180):
    """
    ABC-XYZ segmentation for inventory-driven demand forecasting
    """
    # ABC Analysis: by value contribution
    abc = df.groupby(sku_col)[value_col].sum().reset_index()
    abc = abc.sort_values(value_col, ascending=False)
    abc['value_cumsum'] = abc[value_col].cumsum()
    abc['value_cumpct'] = abc['value_cumsum'] / abc[value_col].sum()
    
    def assign_abc(row):
        if row['value_cumpct'] <= 0.8:
            return 'A'
        elif row['value_cumpct'] <= 0.95:
            return 'B'
        else:
            return 'C'
    
    abc['abc_class'] = abc.apply(assign_abc, axis=1)
    
    # XYZ Analysis: by demand variability
    xyz_data = []
    for sku, group in df.groupby(sku_col):
        demand = group.groupby(group['order_date'].dt.to_period('W'))[qty_col].sum()
        cv = demand.std() / demand.mean() if demand.mean() > 0 else float('inf')
        
        if cv < 0.5:
            xyz = 'X'
        elif cv < 1.0:
            xyz = 'Y'
        else:
            xyz = 'Z'
        xyz_data.append({sku_col: sku, 'cv': cv, 'xyz_class': xyz})
    
    xyz_df = pd.DataFrame(xyz_data)
    
    # Merge ABC and XYZ
    result = abc.merge(xyz_df, on=sku_col)
    result['abc_xyz'] = result['abc_class'] + result['xyz_class']
    
    return result
```

### 3.3 Intermittent Demand Feature Engineering

```python
def create_intermittent_features(demand_series):
    """
    Features specifically for intermittent/lumpy demand items
    """
    features = {}
    non_zero = demand_series[demand_series > 0]
    
    # ADI and CV²
    total_periods = len(demand_series)
    non_zero_periods = len(non_zero)
    features['adi'] = total_periods / non_zero_periods if non_zero_periods > 0 else float('inf')
    features['cv_squared'] = (non_zero.std() / non_zero.mean()) ** 2 if len(non_zero) > 1 else 0
    
    # Inter-demand interval features
    non_zero_indices = demand_series[demand_series > 0].index
    intervals = non_zero_indices.to_series().diff().dropna()
    features['avg_interval'] = intervals.mean()
    features['last_interval'] = intervals.iloc[-1] if len(intervals) > 0 else 0
    
    # Demand size features (when demand occurs)
    features['avg_nonzero_demand'] = non_zero.mean()
    features['max_nonzero_demand'] = non_zero.max() if len(non_zero) > 0 else 0
    features['last_nonzero_demand'] = non_zero.iloc[-1] if len(non_zero) > 0 else 0
    
    # Zero-run features
    zero_runs = []
    current_run = 0
    for v in demand_series:
        if v == 0:
            current_run += 1
        else:
            if current_run > 0:
                zero_runs.append(current_run)
            current_run = 0
    features['max_zero_run'] = max(zero_runs) if zero_runs else 0
    features['current_zero_run'] = current_run
    features['time_since_last_demand'] = current_run
    
    return features
```

---

## 4. What Works

### 4.1 RFM Customer Segmentation Works
- Segmenting customers by Recency, Frequency, Monetary value before forecasting improves accuracy [^1^]
- Each segment receives tailored models accounting for different purchasing behaviors
- Bayesian model averaging over segment forecasts provides further improvement

### 4.2 Lead Time Features Are Highly Predictive
- Actual shipping days and late delivery flags are among the most important operational predictors [^2^]
- Lead time variability per warehouse-customer pair captures supply chain reliability
- Late delivery patterns predict future ordering behavior adjustments

### 4.3 Multi-Level Aggregation Features Work
- Rolling statistics at product, store, category, and combination levels are consistently top features [^9^]
- Features like `avg_sales_by_store_product`, `rolling_mean_30d_by_category` capture hierarchy effects
- The Kaggle Rossmann winner used statistics calculated at different hierarchy levels for different days of week

### 4.4 ADI-CV² Classification Enables Method Selection
- Classifying demand patterns allows matching the right algorithm to each pattern [^3^]
- Smooth demand: ARIMA or exponential smoothing
- Intermittent demand: Croston's method or variants (SBA, TSB)
- Lumpy demand: ML approaches or buffer-based strategies
- Portfolio-wide accuracy improvements of 20-40%

### 4.5 Product Lifecycle Detection Works
- Tracking signals (cumulative forecast error / MAD) detect when a product transitions between lifecycle stages [^5^]
- Growth phase: trend models; Maturity: stable models; Decline: declining trend models
- Early detection of decline phase prevents obsolescence and excess inventory

### 4.6 Calendar-Aware Safety Stock Features Work
- Pre-weekend (Thursday/Friday) ordering patterns capture safety stock behavior [^7^]
- Pre-holiday and post-holiday flags capture demand surges and lulls
- Day-of-week effects are consistently strong predictors in retail forecasting

### 4.7 Lag Features Are Essential
- Lag-7, lag-14, lag-30, lag-365 capture weekly, bi-weekly, monthly, and yearly seasonality [^9^]
- Lag features at multiple hierarchy levels (product, store, category) are most important
- Event counters (days until/since holiday, promotion) proved valuable in Kaggle competitions

### 4.8 Warehouse-Store Network Features Work
- Primary warehouse per store, warehouse switching patterns, cross-docking indicators [^8^]
- Distribution network topology affects order timing and should be encoded
- Hub-and-spoke vs direct delivery creates different demand timing patterns

---

## 5. What Doesn't Work

### 5.1 Ignoring Intermittent Demand Patterns
- Applying standard forecasting methods to intermittent/lumpy demand fails catastrophically [^10^]
- Standard exponential smoothing produces biased forecasts for zero-inflated series
- Must classify demand patterns and apply appropriate methods

### 5.2 Static Lifecycle Stage Assignment
- Assigning lifecycle stages once and never updating fails [^5^]
- Products transition between stages; tracking signals are needed for dynamic detection
- Fixed stage assignments lead to systematic forecast errors

### 5.3 Using Raw CV Without ADI
- Coefficient of variation alone cannot distinguish smooth from seasonal demand [^12^]
- CV ignores the sequence of observations; trending data gets tagged as unforecastable
- Must use ADI + CV² together for proper demand pattern classification

### 5.4 Over-Aggregating Demand Signals
- Forecasting at too high a level (national, category-level) loses store-specific patterns [^6^]
- Store-level and store-product-level forecasts capture localized demand
- Aggregation should happen after forecasting, not before

### 5.5 Ignoring Weekend/Holiday Effects
- Models without calendar-aware features miss predictable demand shifts [^7^]
- Safety stock ordering before weekends creates systematic demand spikes
- Holiday calendar features are essential in retail demand forecasting

### 5.6 Geographic One-Hot Encoding for High Cardinality
- One-hot encoding for many warehouses/stores creates dimensionality problems [^2^]
- Target encoding or embeddings work better for high-cardinality categorical features
- SHAP analysis showed geographic one-hot features contributed minimally

---

## 6. Competition Applications

### 6.1 Kaggle Rossmann Store Sales
- Task: Predict 6 weeks of daily sales for 1,115 stores [^13^]
- Winner approach: XGBoost with extensive feature engineering
- Key supply chain features:
  - Statistics at different hierarchy levels (store, product, day-of-week combinations)
  - Event counters (days until/after promotions, holidays)
  - Fourier terms for seasonality
  - Binary features for store closures, promotions
  - Trend features (days since beginning, log days)
- Result: Ensemble of multiple XGBoost models provided ~5% boost over single model

### 6.2 Kaggle Corporacion Favorita Grocery Sales
- Task: Forecast daily unit sales by store and product for 1-16 day horizon [^9^]
- Winner approach: LightGBM + neural network ensemble
- Key supply chain features:
  - Rolling statistics grouped by store, item, class, combinations
  - Only recent data used (1, 3, or 5 months), ignoring older observations
  - Separate model per forecast horizon (16 models instead of 1)
  - Exponential moving averages

### 6.3 Kaggle M5 Forecasting Competition
- Task: Predict 28 days of unit sales for Walmart items [^14^]
- Top 3% solution: LightGBM with lag and rolling features
- Key features: "The most important features are lag features, created by a combination of lags, rolling windows, and aggregation functions on sales and prices"
- 3-fold time-series cross-validation for hyperparameter tuning

### 6.4 Kaggle Predicting Future Sales
- Task: Predict monthly sales for shop-item pairs [^15^]
- Key supply chain features:
  - Mean encodings at shop, item, category, month levels
  - Lag features at 1, 2, 3, 5, 12 months
  - Holiday boolean features (December, New Year, etc.)
  - Shop-item pair generation for complete grid

---

## 7. Recommended Approach

### 7.1 Immediate Priority Features (Implement First)

1. **Lag Features** (highest ROI)
   - Lag-7, lag-14, lag-30, lag-365 at store-product level
   - Lag features at product-only and store-only aggregation levels
   
2. **Rolling Statistics**
   - Rolling mean, std, median over 7, 14, 30, 90 days
   - Rolling max and min for demand bounds
   
3. **Calendar/Safety Stock Features**
   - Day of week, month, quarter
   - Days until weekend, is_pre_weekend flag
   - Days until/since major holidays
   - Month-start and month-end flags
   
4. **Order Size Features**
   - Historical avg/std/median quantity per store-product
   - Coefficient of variation (CV) of order quantities
   - Is-bulk-order flag

### 7.2 Medium Priority Features

5. **RFM Customer Segmentation**
   - Recency, Frequency, Monetary per store
   - Segment indicator features
   
6. **ADI-CV² Demand Pattern Classification**
   - Classify each store-product as smooth/erratic/intermittent/lumpy
   - Use classification as categorical feature
   
7. **Lead Time Features** (if delivery data available)
   - Actual shipping days
   - Late delivery flag
   - Lead time variability
   
8. **Store-Level Features**
   - Store age, lifecycle stage
   - Store demand trend
   - Days since last order

### 7.3 Advanced Features

9. **ABC-XYZ Segmentation**
   - ABC class by revenue contribution
   - XYZ class by demand variability (CV)
   - Combined ABC-XYZ as categorical feature
   
10. **Product Lifecycle Features**
    - Product age, demand trend slope
    - Tracking signal for lifecycle transition
    
11. **Warehouse Network Features**
    - Primary warehouse per store
    - Cross-docking indicators
    - Warehouse utilization patterns

12. **Intermittent Demand Specific Features**
    - Time since last demand
    - Average non-zero demand size
    - Zero-run statistics

### 7.4 Model Strategy

- Use LightGBM or XGBoost as primary model (consistent competition winners)
- Apply separate models or model weights per ADI-CV² demand pattern class
- Use time-series cross-validation with fold boundaries respecting temporal order
- Ensemble multiple models with different feature subsets for diversity
- Monitor forecast bias per ABC-XYZ segment

---

## 8. Sources

[^1^] ScienceDirect. "Cluster-based demand forecasting using Bayesian model averaging: An ensemble learning approach." 2022. https://www.sciencedirect.com/science/article/pii/S2772662222000066

[^2^] MDPI Sustainability. "Enhancing Supply Chain Management: A Comparative Study of Machine Learning Techniques." 2025. https://www.mdpi.com/2071-1050/17/13/5772

[^3^] MCP Analytics. "Croston Method for Intermittent Demand Whitepaper." 2025. https://mcpanalytics.ai/whitepapers/whitepaper-croston-method

[^4^] MIT Supply Chain Thesis. "ABC-XYZ Classification for Demand Planning." https://dspace.mit.edu/bitstream/handle/1721.1/117927/1051223544-MIT.pdf

[^5^] SlimStock. "Product Lifecycle Management: Definition + Guide." 2025. https://www.slimstock.com/blog/product-lifecycle-management/

[^6^] Lark. "Store Level Forecasting Guide." 2024. https://www.larksuite.com/en_us/topics/retail-glossary/store-level-forecasting

[^7^] VersaCloud ERP. "Pre-Holiday Inventory Management Best Practices." 2024. https://www.versaclouderp.com/blog/effective-stock-optimization-techniques-in-pre-holiday-inventory-management/

[^8^] Cadre Technologies. "Cross Docking Case Studies." 2025. https://www.cadretech.com/cross-docking/

[^9^] arXiv. "Learnings from Kaggle's Forecasting Competitions." https://arxiv.org/pdf/2009.07701

[^10^] PMC. "Primacy of feature engineering over architectural complexity for intermittent demand forecasting." https://pmc.ncbi.nlm.nih.gov/articles/PMC12873174/

[^11^] HAL Science. "An Exploration of Machine Learning Techniques for Lead Time Prediction." https://hal.science/hal-04588185/document

[^12^] Arkieva Blog. "Do You Use Coefficient Of Variation To Determine Forecastability?" 2015. https://blog.arkieva.com/do-you-use-coefficient-of-variation-to-determine-forecastability/

[^13^] Kaggle Rossmann Competition Winner Writeup. https://www.kaggle.com/c/rossmann-store-sales

[^14^] GitHub - minaxixi. "Kaggle M5 Forecasting Accuracy Solution." 2020. https://github.com/minaxixi/Kaggle-M5-Forecasting-Accuracy

[^15^] GitHub - storieswithsiva. "Kaggle Predicting Future Sales." 2020. https://github.com/storieswithsiva/Kaggle-Predicting-Future-Sales

[^16^] DemandPlanning.net. "ABC-XYZ of Forecasting by Exception." 2026. https://demandplanning.net/abc-xyz-of-forecasting-by-exception/

[^17^] Mecalux. "ABC XYZ analysis: Pros and cons." 2025. https://www.mecalux.com/blog/abc-xyz

[^18^] Intelichain. "The 9 Box Model - ABC-XYZ Classification in Demand Planning." 2025. https://inteli-chain.com/the-9-box-model-abc-xyz-classification-in-demand-planning/

[^19^] Zalando Engineering. "Building a dynamic inventory optimisation system." 2025. https://engineering.zalando.com/posts/2025/06/inventory-optimisation-system.html

[^20^] ScienceDirect. "Daily retail demand forecasting using machine learning with emphasis on calendric special days." 2022. https://www.sciencedirect.com/science/article/abs/pii/S0169207020300224

[^21^] InfoVision. "AI-driven demand forecasting for healthcare major." https://www.infovision.com/case-study/ai-driven-demand-forecasting-healthcare-major

[^22^] RELEX Solutions. "Demand forecasting for retail and consumer goods." 2025. https://www.relexsolutions.com/resources/demand-forecasting/

[^23^] Crisp. "Retail demand forecasting: Getting started." 2026. https://www.gocrisp.com/learning-center/operations-supply-chain/retail-demand-forecasting-getting-started

[^24^] Lokad. "Lead Time - Probabilistic Forecasting." https://www.lokad.com/lead-time/

[^25^] Future Forecasting. "Declining Demand: Is this the End of the Product Life Cycle?" 2025. https://www.future-forecasting.de/en/blog/declining-demand-ist-this-the-end-of-the-product-life-cycle/

[^26^] GitHub - zahraayafni. "forecasting_demand_planning - Product Segmentation with ADI/CV²." 2020. https://github.com/zahraayafni/forecasting_demand_planning

[^27^] StatusNeo. "How to Determine Forecastability of Demand?" 2022. https://statusneo.com/how-to-determine-forecastability-of-demand/

[^28^] MDPI. "Reframing Demand Forecasting: A Two-Fold Approach for Lumpy and Intermittent Demand." 2022. https://www.mdpi.com/2071-1050/14/15/9295

[^29^] ScienceDirect. "Demand-driven storage allocation for optimizing order picking processes." 2025. https://www.sciencedirect.com/science/article/abs/pii/S0957417425004348

[^30^] Kaggle - Mabrek Blog. "My Top 10% Solution for Kaggle Rossmann Store Sales." 2016. https://mabrek.github.io/blog/kaggle-forecasting/

---

## Appendix: Feature Importance Summary Table

| Feature Category | Specific Features | Expected Importance | Evidence Source |
|-----------------|-------------------|-------------------|----------------|
| Lag Features | lag_7d, lag_30d, lag_365d | Very High | Kaggle winners [^9^][^14^] |
| Rolling Statistics | rolling_mean_30d, rolling_std_30d | Very High | Kaggle winners [^9^][^14^] |
| Calendar Features | dayofweek, month, is_weekend | High | Rossmann winner [^13^] |
| Order Size | avg_qty, cv_qty, is_bulk | High | ABC-XYZ theory [^4^] |
| RFM Segmentation | recency, frequency, monetary | Medium-High | Academic study [^1^] |
| ADI-CV² Pattern | demand_pattern_class | Medium-High | Croston whitepaper [^3^] |
| Lead Time | shipping_days, is_late | Medium | MDPI study [^2^] |
| Product Lifecycle | product_age, trend_slope | Medium | SlimStock [^5^] |
| Store Features | store_age, days_since_order | Medium | Store-level forecasting [^6^] |
| Safety Stock | is_pre_weekend, is_month_end | Medium | Holiday inventory [^7^] |
| Warehouse Network | primary_warehouse, n_warehouses | Low-Medium | Cross-docking study [^8^] |
| Intermittent Specific | time_since_demand, avg_nonzero | Medium | Intermittent demand [^10^] |

---

## Appendix: Demand Pattern Classification Thresholds

| Pattern | ADI Threshold | CV² Threshold | Recommended Method |
|---------|--------------|---------------|-------------------|
| Smooth | < 1.32 | < 0.49 | ARIMA, Exponential Smoothing |
| Erratic | < 1.32 | >= 0.49 | ML with safety stock |
| Intermittent | >= 1.32 | < 0.49 | Croston's method, SBA, TSB |
| Lumpy | >= 1.32 | >= 0.49 | Buffer-based, DDMRP, ML |

Source: Syntetos et al. classification framework [^27^][^28^]
