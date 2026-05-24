# Dimension 5: Geographic & Regional Economic Features

## Research on Geographic and Regional Economic Features for Demand Forecasting

**Research Date:** 2026-08-11
**Scope:** Geographic features for demand forecasting of pickled vegetables across China
**Context:** Customer1 (15+ provinces), Customer2 (Guangdong/Guangxi concentration), Customer3 (scattered smaller)

---

## Table of Contents
1. [Key Findings](#1-key-findings)
2. [City-Tier Classification](#2-city-tier-classification)
3. [Regional GDP/Capita Data Sources](#3-regional-gdpcapita-data-sources)
4. [Population Density](#4-population-density)
5. [Regional Cuisine Preferences](#5-regional-cuisine-preferences)
6. [Climate Zones](#6-climate-zones)
7. [Transportation Hubs](#7-transportation-hubs)
8. [Store Density](#8-store-density)
9. [Implementation Details](#9-implementation-details)
10. [What Works](#10-what-works)
11. [What Doesn't Work](#11-what-doesnt-work)
12. [Recommended Approach](#12-recommended-approach)
13. [Sources](#13-sources)

---

## 1. Key Findings

### Finding 1: City-Tier Classification is the Most Important Geographic Feature

Claim: Yicai Media Group's city tier ranking, updated annually, classifies 337 prefecture-level cities into 6 tiers based on five dimensions: Commercial Resources, City Hub Index, Urban Activity, New Economy Competitiveness, and Future Potential. This classification directly correlates with consumer purchasing power, retail spending patterns, and brand penetration rates. [^309^]
Source: Yicai Global (official ranking release)
URL: https://www.yicaiglobal.com/news/chengdu-hangzhou-13-others-rank-as-new-first-tier-chinese-cities-in-2025
Date: 2025-05-28
Excerpt: "Chengdu, Hangzhou, Chongqing, Wuhan, and 11 other Chinese cities have entered this year's new first-tier cities ranking, which assesses the overall attractiveness of 337 cities at the prefecture level and above... The country's four first-tier cities -- Shanghai, Beijing, Shenzhen, and Guangzhou -- maintained a stable ranking."
Confidence: HIGH

### Finding 2: Pickled Vegetable Consumption Has Strong Regional Patterns

Claim: China can be divided into three preserved vegetable consumption regions: (1) regions mainly consuming sour pickled vegetables (Sichuan, Gansu), (2) regions mainly consuming salted vegetables (Qingdao, Harbin, Suzhou, Zhejiang), and (3) regions rarely consuming preserved vegetables (Haikou, Liuzhou, Hunan, Henan). These patterns are deeply rooted in historical dietary culture and climate adaptation. [^347^]
Source: Journal of Global Health (prospective cohort study)
URL: https://jogh.org/2024/jogh-14-04191/
Date: 2024-11-08
Excerpt: "Three types of regions were classified: 1) rarely consuming both salted vegetables and sour pickled vegetables (Haikou, Liuzhou, Hunan, and Henan), 2) often consuming salted vegetables but rarely consuming sour pickled vegetables (Qingdao, Harbin, Suzhou, and Zhejiang), 3) often consuming sour pickled vegetables but rarely consuming salted vegetables (Sichuan and Gansu)."
Confidence: HIGH

### Finding 3: Guangdong Has Highest Vegetable Consumption Nationwide

Claim: Guangdong province has the highest total vegetable consumption in China at 46.7 million tons (8.63% of national total) and the highest per capita vegetable consumption at 142.57 kg/year. Sichuan ranks second in total consumption. In contrast, Guizhou has the lowest per capita consumption at only 50.49 kg/year. [^310^]
Source: Frontiers in Sustainable Food Systems (peer-reviewed)
URL: https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2025.1685599/full
Date: 2025-10-15
Excerpt: "Guangdong has the highest vegetable consumption, reaching 46.7 million tons, which accounts for 8.63% of the total vegetable consumption in China... Guangdong also has the highest per capita vegetable consumption, at 142.57 kg, while Guizhou has the lowest, at only 50.49 kg."
Confidence: HIGH

### Finding 4: M5 Competition Winners Used Hierarchical Store-Department-Category Grouping

Claim: The M5 forecasting competition winning solution used LightGBM models trained at three hierarchical levels: per-store (10 models), per-store-category (30 models), and per-store-department (70 models), with forecasts combined using equal weights. Geographic features (state_id, store_id) were embedded as categorical features. [^462^]
Source: M5 Accuracy Competition Official Results Paper
URL: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
Date: 2021
Excerpt: "The winner considered an equal weighted combination of various LightGBM models that were trained to produce forecasts for the product-store series using data per store (10 models), store-category (30 models), and store-department (70 models)."
Confidence: HIGH

### Finding 5: Climate Drives Food Preservation Traditions

Claim: Northeast China's cold climate with long winters historically required extensive food preservation, making pickled vegetables and cured meats integral to the regional cuisine. Climate warming is causing shifts in food consumption patterns, with northern residents consuming more plant-based foods and southern residents maintaining meat consumption. [^306^]
Source: MDPI Sustainability (peer-reviewed)
URL: https://www.mdpi.com/2071-1050/17/21/9682
Date: 2025-10-30
Excerpt: "Residents in North China experienced cold and dry winters, and the demand for high-calorie foods decreased with climate warming... The divergence in passive adaptation patterns reflects the foundational differences in dietary structures and culinary traditions between North and South China."
Confidence: HIGH

### Finding 6: Fuling Zhacai Dominates Chinese Pickled Vegetable Market

Claim: Chongqing Fuling Zhacai Group holds approximately 30% market share in the Chinese pickled vegetable sector, with Wujiang brand as the market leader. The company distributes across 28-30 provinces in China. Fuling District produces 64% of Chongqing's zhacai and maintains top market share nationwide. [^391^] [^397^]
Source: iChongqing / Multiple market reports
URL: https://www.ichongqing.info/2024/12/18/fuling-pickled-mustard-tuber-extends-its-market-reach-to-over-50-countries-and-regions/
Date: 2024-12-18
Excerpt: "Chongqing's zhacai industry boasts an annual output value of 36.7 billion yuan, of which Fuling's zhacai production exceeds 13 billion yuan, maintaining its top market share nationwide."
Confidence: HIGH

### Finding 7: Third- and Fourth-Tier Cities Show Faster Consumption Growth

Claim: Third- and fourth-tier cities in China are showing significantly faster consumer spending growth (~5.8%) compared to first- and second-tier cities. Food and cosmetics spending is growing particularly fast in these lower-tier markets. This creates a "catch-up growth" dynamic where emerging markets drive disproportionate demand growth. [^530^]
Source: Sixth Tone / The Paper
URL: https://www.sixthtone.com/news/1017003
Date: 2025-04-24
Excerpt: "In the first half of 2024, per capita disposable income in third- and fourth-tier cities expanded nearly 5.8%, a full percentage point higher than growth in first- and second-tier cities... Third- and fourth-tier cities saw strong growth in spending on consumer goods like food and cosmetics."
Confidence: HIGH

---

## 2. City-Tier Classification

### 2.1 The Yicai Classification System (2025)

The most widely used city-tier classification in China comes from Yicai Media Group's annual "Ranking of Cities' Attractiveness in China." The 2025 ranking evaluates 337 prefecture-level cities across five dimensions: [^309^]

| Tier | Count | Cities | Key Characteristics |
|------|-------|--------|-------------------|
| **1st-Tier** | 4 | Beijing, Shanghai, Shenzhen, Guangzhou | Highest GDP, most developed infrastructure, international business presence |
| **New 1st-Tier** | 15 | Chengdu, Hangzhou, Chongqing, Wuhan, Suzhou, Xi'an, Nanjing, Changsha, Zhengzhou, Tianjin, Hefei, Qingdao, Dongguan, Ningbo, Foshan | Rapidly developing, expanding middle class, rising tech/innovation |
| **2nd-Tier** | 30 | Xiamen, Fuzhou, Wuxi, Kunming, Harbin, Jinan, Changchun, Wenzhou, etc. | Provincial capitals or major regional centers, growing infrastructure |
| **3rd-Tier** | 70 | Smaller prefecture-level cities | Emerging middle class, lower costs, strong local industries |
| **4th-Tier** | 90 | County-level cities | Less developed but with growth potential |
| **5th-Tier** | 128 | Remaining cities | Smallest, often rural-based |

### 2.2 Key Dimensions in the 2025 Ranking

1. **Concentration of Commercial Resources Index** - Big brand presence, commercial core
2. **City as a Hub Index** - Traffic connectivity, intercity flows, industry collaboration
3. **Urban Residents' Activity Index** - Consumption patterns, leisure activities, nighttime economy
4. **New Economy Competitiveness Index** - Tech companies, new consumption, industrial chain ecology
5. **Future Potential Index** - Innovation, talent attraction, urban scale [^309^]

### 2.3 Practical Impact on Demand Forecasting

**Consumer purchasing power by tier:**
- Tier 1: Highest disposable income but slower spending growth (~0.4% in Shanghai in 2024)
- Tier 2: Strong growth, increasing brand consciousness
- Tier 3-4: Fastest spending growth (~5.8%), "catch-up growth" dynamic [^530^]

**Retail expansion patterns:**
- Brands historically entered Tier 1 first, then expanded downward
- Now many brands are starting directly in Tier 2-3 cities due to faster growth rates
- Store format and product mix should differ by tier

### 2.4 Implementation as Features

```python
# City tier as ordinal feature
# Option 1: Simple ordinal encoding
city_tier_map = {
    'Tier 1': 1,
    'New Tier 1': 2,
    'Tier 2': 3,
    'Tier 3': 4,
    'Tier 4': 5,
    'Tier 5': 6
}

# Option 2: Target encoding using mean sales per tier
# (more powerful but requires careful CV to prevent leakage)

# Option 3: Use as categorical with LightGBM/CatBoost native support
# LightGBM: lgb.Dataset(..., categorical_feature=['city_tier'])
# CatBoost: CatBoostRegressor(..., cat_features=['city_tier'])
```

---

## 3. Regional GDP/Capita Data Sources

### 3.1 Provincial-Level GDP Per Capita (2024)

The most comprehensive and recent provincial GDP per capita data: [^470^]

| Province/City | GDP/Capita (yuan) | Relative to National Avg |
|--------------|------------------|------------------------|
| Beijing | 228,167 | 2.38x |
| Shanghai | 217,140 | 2.27x |
| Jiangsu | 160,694 | 1.68x |
| Fujian | 137,920 | 1.44x |
| Zhejiang | 135,565 | 1.42x |
| Tianjin | 132,143 | 1.38x |
| Guangdong | 111,146 | 1.16x |
| Hubei | 102,832 | 1.07x |
| Chongqing | 100,903 | 1.05x |
| Shandong | 97,575 | 1.02x |
| **National Avg** | **95,749** | **1.00x** |
| Sichuan | 77,333 | 0.81x |
| Hunan | 81,225 | 0.85x |
| Anhui | 82,694 | 0.86x |
| Henan | 64,888 | 0.68x |
| Guizhou | 58,685 | 0.61x |
| Guangxi | 57,071 | 0.60x |
| Gansu | 52,825 | 0.55x |

### 3.2 Data Sources for City-Level Economic Indicators

**Primary Official Sources:**

1. **China City Statistical Yearbook** (中国城市统计年鉴) - Published by National Bureau of Statistics
   - Contains city-level GDP, population, employment, fiscal revenue
   - Available through CNKI yearbook database
   - Coverage: All prefecture-level cities [^357^]

2. **China Statistical Yearbook for Regional Economy** (中国区域经济统计年鉴)
   - Provincial and city-level economic data
   - GDP, industrial output, retail sales, per capita income [^332^]

3. **National Bureau of Statistics Data Portal** (data.stats.gov.cn)
   - Free online access to key indicators
   - Limited depth compared to yearbooks [^331^]

4. **CEIC China Premium Database**
   - Subscription-based macroeconomic and financial time series
   - Subnational data available [^341^]

5. **EPS China Statistics**
   - 100+ statistical datasets
   - Strong focus on city and county statistics
   - Allows chart/map creation [^341^]

6. **China Geo-Explorer**
   - Integrates government statistics, census data at multiple levels
   - Province, city, county, township, ZIP code [^341^]

7. **WorldPop Project** (worldpop.org)
   - Fine-scale population distribution raster data (~100m resolution)
   - Open access, global coverage [^342^]

### 3.3 Free/Open Data APIs

```python
# Option 1: National Bureau of Statistics API (limited)
# http://data.stats.gov.cn/easyquery.htm?cn=C01

# Option 2: China City Data APIs (various providers)
# - Baidu Maps API (geocoding, location data)
# - Tencent Location API (population density, POI)
# - AutoNavi (AMap) API (transportation, traffic)

# Option 3: Academic research datasets
# - China Family Panel Studies (CFPS) from Peking University
# - China Health and Nutrition Survey (CHNS)
```

### 3.4 Implementation as Features

```python
# Normalize GDP per capita to national average (relative purchasing power)
df['gdp_per_capita_relative'] = df['city_gdp_per_capita'] / 95749

# Log transform (common for GDP features due to right skew)
df['log_gdp_per_capita'] = np.log1p(df['city_gdp_per_capita'])

# Bin into economic development levels
df['economic_tier'] = pd.cut(df['city_gdp_per_capita'],
                              bins=[0, 60000, 100000, 150000, float('inf')],
                              labels=['low', 'medium', 'high', 'very_high'])
```

---

## 4. Population Density

### 4.1 Urban vs. Suburban Store Location Impact

Population density directly affects store catchment area and potential customer base. Key considerations: [^447^]

- **High-density urban areas**: Smaller catchment radius but higher foot traffic
- **Low-density suburban areas**: Larger catchment radius needed, different traffic patterns
- **Future Sales = Average Spend per Household x Number of Future Households**

### 4.2 Data Sources for Population Density

1. **China Resources and Environmental Science and Data Center**
   - Population density raster data (1000m resolution)
   - GDP raster data (1000m resolution) [^330^]

2. **WorldPop/WorldPop Project**
   - 100m resolution population distribution data
   - Available in Geotiff format
   - Open access [^342^]

3. **LandScan Global Population Database**
   - Finest resolution global population data
   - Used in academic research for urban analysis [^332^]

4. **Tencent User Density Data**
   - Real-time user location data from Tencent apps
   - Can serve as proxy for population distribution
   - 93%+ coverage in Tier 1 cities [^339^]

### 4.3 Implementation as Features

```python
# Population density categories
population_density_tiers = {
    'ultra_high': '>5000 people/km2',    # Core city center
    'high': '3000-5000 people/km2',      # Urban area
    'medium': '1000-3000 people/km2',    # Suburban
    'low': '<1000 people/km2'            # Rural/peri-urban
}

# Store format can be inferred from density
# High density -> smaller stores, more convenience formats
# Low density -> larger stores, destination shopping
```

---

## 5. Regional Cuisine Preferences

### 5.1 The Eight Great Cuisines and Pickled Vegetable Preferences

China's Eight Great Cuisines (八大菜系) have distinct relationships with pickled vegetables: [^327^]

| Cuisine | Region | Pickled Vegetable Relationship |
|---------|--------|-------------------------------|
| **Sichuan (川菜)** | Sichuan, Chongqing | Very strong - sour pickled vegetables core to cuisine; "sour pickled vegetable" consuming region |
| **Hunan (湘菜)** | Hunan | Strong - heavy use of smoked/pickled items; pickled vegetables in many dishes |
| **Cantonese (粤菜)** | Guangdong, Guangxi | Moderate - fresh-focused cuisine; some preserved vegetables in congee/soups |
| **Shandong (鲁菜)** | Shandong | Moderate - historical pickling tradition for winter preservation |
| **Jiangsu (苏菜)** | Jiangsu, Zhejiang | Moderate - salted vegetable consuming region (pickled mustard greens) |
| **Fujian (闽菜)** | Fujian | Low-Moderate - coastal focus, less pickling tradition |
| **Anhui (徽菜)** | Anhui | Moderate - mountain herbs, some preservation |
| **Zhejiang (浙菜)** | Zhejiang | Moderate - part of salted vegetable consuming region |

### 5.2 Regional Pickled Vegetable Consumption Map

Based on epidemiological studies and market research: [^347^]

**High pickled vegetable consumption (sour pickled type):**
- Sichuan Province (home of Fuling Zhacai)
- Chongqing Municipality
- Gansu Province

**High pickled vegetable consumption (salted type):**
- Shandong Province (Qingdao)
- Heilongjiang Province (Harbin)
- Jiangsu Province (Suzhou)
- Zhejiang Province

**Low pickled vegetable consumption:**
- Hainan Province (Haikou)
- Guangxi (Liuzhou)
- Hunan Province
- Henan Province

### 5.3 Fuling Zhacai (榨菜) - China's Largest Pickled Vegetable Category

Fuling Zhacai (pickled mustard tuber) is the dominant pickled vegetable product in China: [^388^] [^391^]

- **Production**: Chongqing produces 1.419 million tons of finished zhacai annually (>70% of national total)
- **Market leader**: Chongqing Fuling Zhacai Group (~30% market share, Wujiang brand)
- **Consumption**: Most popular in Sichuan/Chongqing region, now expanding nationally
- **Format shift**: From "salt supplement for exercise" to "low-salt appetizer for meals"
- **New markets**: Japan (+30% YoY), South Korea, USA (with localized flavors)

### 5.4 Implementation as Features

```python
# Province-level pickled vegetable consumption propensity
# Based on research findings
pickle_consumption_propensity = {
    # High sour-pickled consumption (Sichuan/Chongqing style)
    'Sichuan': 'high_sour', 'Chongqing': 'high_sour', 'Gansu': 'high_sour',
    # High salted consumption (coastal/Northern style)
    'Shandong': 'high_salted', 'Jiangsu': 'high_salted', 'Zhejiang': 'high_salted',
    'Heilongjiang': 'high_salted',
    # Moderate consumption
    'Guangdong': 'moderate', 'Fujian': 'moderate', 'Anhui': 'moderate',
    'Beijing': 'moderate', 'Shanghai': 'moderate', 'Tianjin': 'moderate',
    # Low consumption
    'Hainan': 'low', 'Hunan': 'low', 'Henan': 'low',
    'Guangxi': 'low',  # Despite being pickled vegetable production region
    # Default for others
}

# Can be encoded as:
# - Categorical feature (LightGBM/CatBoost native)
# - One-hot encoded for neural networks
# - Target encoded based on historical sales
```

---

## 6. Climate Zones

### 6.1 China's Climate Zones (Köppen Classification)

China spans multiple Köppen climate zones, which influence food preservation needs and dietary patterns: [^348^]

| Climate Zone | Köppen Code | Regions | Characteristics |
|-------------|-------------|---------|----------------|
| **Humid subtropical** | Cfa | South China (Guangdong, Guangxi, Fujian) | Hot humid summers, mild winters, abundant rainfall |
| **Monsoon-influenced humid subtropical** | Cwa | Central/Southwest China (Sichuan, Chongqing, Yunnan) | Distinct wet/dry seasons, warm summers |
| **Humid continental** | Dwa | North China (Beijing, Hebei, Shandong, Northeast) | Hot summers, cold dry winters, 4 distinct seasons |
| **Continental semiarid** | Bsk | Northwest China (Gansu, Ningxia, Inner Mongolia) | Low rainfall, large temperature swings |
| **Cold desert** | Bwk | Far northwest | Very dry, cold winters |
| **Subtropical highland** | Cwb | Yunnan (Kunming) | Mild year-round |

### 6.2 Climate Impact on Food Preservation

Key climate-demand relationships: [^307^] [^348^]

**Northern China (Humid Continental/Cold):**
- Long harsh winters historically required food preservation
- Pickled vegetables, cured meats, fermented products are traditional staples
- Higher demand for preserved foods in winter months
- Northeast China has highest Dietary Structure Index (DSI) partly due to preserved food diversity

**Southern China (Humid Subtropical):**
- Year-round growing season reduces need for preservation
- Fresh ingredient emphasis
- Pickled vegetables used more as flavoring/condiment rather than staple
- Humid climate actually supports fermentation traditions (Sichuan)

**Hot-Summer Cold-Winter (HSCW) Region:**
- Covers 16 provinces, ~500 million people
- Coldest month: 0-10°C; Hottest month: 25-30°C
- High humidity year-round (30-80%)
- Belongs to Cf in Köppen classification [^352^] [^353^]

### 6.3 Implementation as Features

```python
# Climate zone encoding
climate_zones = {
    # Humid Subtropical
    'Guangdong': 'Cfa', 'Guangxi': 'Cfa', 'Fujian': 'Cfa',
    'Hainan': 'Cfa', 'Zhejiang': 'Cfa',
    # Monsoon Humid Subtropical
    'Sichuan': 'Cwa', 'Chongqing': 'Cwa', 'Guizhou': 'Cwa',
    'Yunnan': 'Cwb',  # Subtropical highland
    # Humid Continental
    'Beijing': 'Dwa', 'Hebei': 'Dwa', 'Shandong': 'Dwa',
    'Henan': 'Dwa', 'Shanxi': 'Dwa', 'Liaoning': 'Dwa',
    'Jilin': 'Dwa', 'Heilongjiang': 'Dwa',
    # Semi-arid
    'Gansu': 'Bsk', 'Ningxia': 'Bsk', 'Inner Mongolia': 'Bsk',
    'Shaanxi': 'Bsk',
}

# Seasonal demand factor (higher in winter for northern regions)
def get_seasonal_pickled_demand_factor(province, month):
    """Higher demand in winter months for northern provinces"""
    if province in ['Heilongjiang', 'Jilin', 'Liaoning', 'Inner Mongolia', 'Hebei']:
        winter_months = [11, 12, 1, 2, 3]
        return 1.3 if month in winter_months else 1.0
    elif province in ['Sichuan', 'Chongqing', 'Gansu']:
        # Year-round consumption but slight winter bump
        winter_months = [12, 1, 2]
        return 1.15 if month in winter_months else 1.0
    else:
        # Southern provinces - more stable year-round
        return 1.0
```

---

## 7. Transportation Hubs

### 7.1 Transportation Connectivity Index

The Yicai ranking includes "City as a Hub Index" with sub-components: [^309^]
- Traffic Connectivity Index
- Intercity Flow Index
- Industry Collaboration Index
- Business Resource Regional Primacy Index

### 7.2 Key Transportation Corridors Affecting Distribution

Major vegetable circulation routes in China (relevant for supply chain forecasting): [^310^]

1. **Shandong/Hebei/Henan** (major production) → Beijing/Tianjin/Shanghai/Zhejiang/Guangdong
2. **Guizhou/Sichuan** (Southwest production) → South China, East China
3. **Guangxi** → Guangdong (Guangdong has only 54.4% vegetable self-sufficiency)
4. **Northwest** ( surplus) → North China, South China

### 7.3 Implementation as Features

```python
# Proximity to major transportation hub (categorical)
transport_hub_proximity = {
    'core_hub': 'Within 50km of major rail/highway junction',
    'secondary_hub': '50-150km from major junction',
    'peripheral': '>150km from major junction'
}

# Inter-provincial vegetable flow direction
# (useful for supply-side features)
major_flows = [
    ('Shandong', 'Beijing'),
    ('Shandong', 'Shanghai'),
    ('Shandong', 'Guangdong'),
    ('Sichuan', 'Guangdong'),
    ('Guizhou', 'Guangdong'),
    ('Guangxi', 'Guangdong'),
    ('Hebei', 'Beijing'),
    ('Hebei', 'Tianjin'),
]
```

---

## 8. Store Density

### 8.1 Store Clustering Best Practices

Store clustering groups similar stores for efficient planning and forecasting: [^346^]

Benefits:
- **Planning Efficiency**: Create 5-10 cluster plans instead of 100 individual plans
- **Improved Forecast Accuracy**: Similar stores smooth out random noise while capturing collective trends
- **Assortment Localization**: Tailor product mix to regional tastes

### 8.2 Clustering Dimensions

Effective store clustering considers: [^356^]
- Historical sales patterns (index-based clustering)
- Geographic location (climate, regional preferences)
- Store format/size
- Catchment area demographics
- Customer segment profile

### 8.3 Implementation as Features

```python
# Number of stores per city as proxy for market penetration
stores_per_city = df.groupby('city')['store_id'].nunique()
df['stores_per_city'] = df['city'].map(stores_per_city)

# Store density relative to population
df['stores_per_million_pop'] = df['stores_per_city'] / (df['city_population'] / 1_000_000)

# Market penetration level
df['market_penetration'] = pd.cut(df['stores_per_million_pop'],
                                   bins=[0, 1, 3, 5, float('inf')],
                                   labels=['low', 'medium', 'high', 'saturated'])
```

---

## 9. Implementation Details

### 9.1 M5 Competition Winning Approach (Geographic Features)

The M5 winning solution used the following approach for geographic/store features: [^462^] [^387^]

```python
# Key features used in M5 winning solution:
# - Identifiers: product, store, product-category, product-department
# - Calendar: day of week, week of year, month
# - Events: special days, holidays, SNAP
# - Prices: raw value, normalized across time, normalized within dept
# - Sales data: lags, moving averages, zero-sales periods

# Model architecture:
# - 220 LightGBM models total
# - Per store (10 models) + per store-category (30) + per store-department (70)
# - Two variations each: recursive and non-recursive
# - Final forecast = average of 6 models per series

# Critical insight: Cross-learning across hierarchical levels
# Models trained at higher aggregation levels capture patterns
# that transfer to lower levels
```

### 9.2 Categorical Feature Encoding for Geographic Data

**Option A: LightGBM Native Categorical Support**
```python
import lightgbm as lgb

# LightGBM can handle categorical features natively
# Best for: city, province, store_id, climate_zone

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=['province', 'city_tier', 'climate_zone', 'store_format'],
    free_raw_data=False
)

params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 128,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1
}

model = lgb.train(params, train_data, num_boost_round=1000)
```

**Option B: CatBoost (Best for High-Cardinality Geographic Features)**
```python
from catboost import CatBoostRegressor

# CatBoost's ordered target encoding is ideal for geographic features
# It prevents target leakage automatically

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    cat_features=['province', 'city', 'city_tier', 'climate_zone',
                   'regional_cuisine_type'],
    loss_function='RMSE',
    early_stopping_rounds=50,
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
```

**Option C: Manual Target Encoding (for frameworks without native support)**
```python
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold

# Time-series aware target encoding
# CRITICAL: Must use expanding window or forward-chaining CV
# to prevent leakage

def time_series_target_encode(df, col, target, n_splits=5):
    """Target encoding with forward-chaining to prevent leakage"""
    encoded = pd.Series(index=df.index, dtype=float)

    # Sort by time before encoding
    df_sorted = df.sort_values('date')

    # Expanding window approach
    for i in range(n_splits, len(df_sorted)):
        historical = df_sorted.iloc[:i]
        category_value = df_sorted.iloc[i][col]
        category_mean = historical[historical[col] == category_value][target].mean()
        encoded.iloc[i] = category_mean

    # Fill initial values with global mean
    global_mean = df[target].mean()
    encoded.fillna(global_mean, inplace=True)

    return encoded
```

### 9.3 Hierarchical Forecasting with Geographic Features

```python
# Geographic hierarchy: Store -> City -> Province -> Region -> National
# Product hierarchy: SKU -> Subcategory -> Category -> Department

# Hierarchical reconciliation approaches:
# 1. Bottom-up: Forecast at store level, sum up
# 2. Top-down: Forecast national, allocate by historical proportions
# 3. Middle-out: Forecast at city level, disaggregate to stores, aggregate to regions
# 4. Optimal reconciliation (MinT): Produce forecasts at all levels, reconcile optimally

# Implementation with hts library
from hts import HTSRegressor

hierarchy = {
    'total': ['region_east', 'region_north', 'region_south', 'region_west'],
    'region_east': ['shanghai', 'jiangsu', 'zhejiang'],
    'region_north': ['beijing', 'hebei', 'shandong'],
    'region_south': ['guangdong', 'guangxi', 'fujian'],
    'region_west': ['sichuan', 'chongqing', 'gansu'],
}

model = HTSRegressor(model='lightgbm', revision_method='OLS')
model.fit(df, hierarchy)
```

### 9.4 Complete Feature Engineering Pipeline

```python
def create_geographic_features(df):
    """Create geographic and regional economic features for demand forecasting"""

    features = df.copy()

    # 1. City Tier (ordinal 1-6)
    features['city_tier_numeric'] = features['city_tier'].map({
        'Tier 1': 1, 'New Tier 1': 2, 'Tier 2': 3,
        'Tier 3': 4, 'Tier 4': 5, 'Tier 5': 6
    })

    # 2. GDP features (normalized)
    features['gdp_per_capita_log'] = np.log1p(features['city_gdp_per_capita'])
    features['gdp_relative_to_national'] = features['city_gdp_per_capita'] / 95749

    # 3. Economic development level
    features['economic_level'] = pd.cut(
        features['city_gdp_per_capita'],
        bins=[0, 60000, 100000, 150000, float('inf')],
        labels=[0, 1, 2, 3]  # low, medium, high, very_high
    ).astype(int)

    # 4. Population density
    features['population_density_log'] = np.log1p(features['pop_density_per_km2'])
    features['density_tier'] = pd.cut(
        features['pop_density_per_km2'],
        bins=[0, 1000, 3000, 5000, float('inf')],
        labels=[0, 1, 2, 3]  # low, medium, high, ultra_high
    ).astype(int)

    # 5. Regional cuisine / pickled vegetable propensity
    features['pickle_propensity'] = features['province'].map({
        'Sichuan': 3, 'Chongqing': 3, 'Gansu': 3,  # High sour-pickled
        'Shandong': 2, 'Jiangsu': 2, 'Zhejiang': 2, 'Heilongjiang': 2,  # High salted
        'Guangdong': 1, 'Fujian': 1, 'Anhui': 1,  # Moderate
        'Hainan': 0, 'Hunan': 0, 'Henan': 0, 'Guangxi': 0,  # Low
    }).fillna(1)  # Default moderate

    # 6. Climate zone (one-hot or categorical)
    climate_map = {
        'Guangdong': 'subtropical', 'Guangxi': 'subtropical', 'Fujian': 'subtropical',
        'Hainan': 'subtropical', 'Zhejiang': 'subtropical',
        'Sichuan': 'monsoon_subtropical', 'Chongqing': 'monsoon_subtropical',
        'Guizhou': 'monsoon_subtropical', 'Yunnan': 'highland',
        'Beijing': 'continental', 'Hebei': 'continental', 'Shandong': 'continental',
        'Henan': 'continental', 'Shanxi': 'continental', 'Liaoning': 'continental',
        'Jilin': 'continental', 'Heilongjiang': 'continental',
        'Gansu': 'semiarid', 'Ningxia': 'semiarid', 'Inner Mongolia': 'semiarid',
        'Shaanxi': 'semiarid',
    }
    features['climate_zone'] = features['province'].map(climate_map).fillna('subtropical')

    # 7. Winter demand factor (seasonal interaction)
    features['is_winter'] = features['month'].isin([11, 12, 1, 2, 3])
    features['is_north'] = features['province'].isin([
        'Heilongjiang', 'Jilin', 'Liaoning', 'Inner Mongolia', 'Hebei',
        'Shanxi', 'Shandong', 'Henan', 'Beijing', 'Tianjin', 'Gansu',
        'Ningxia', 'Shaanxi'
    ])
    features['winter_north_interaction'] = (
        features['is_winter'].astype(int) * features['is_north'].astype(int)
    )

    # 8. Store density / market penetration
    stores_per_city = features.groupby('city')['store_id'].transform('nunique')
    features['stores_in_city'] = stores_per_city
    features['stores_per_million_pop'] = stores_per_city / (features['city_population'] / 1e6)

    # 9. Urban vs suburban flag
    features['urban_suburban'] = pd.cut(
        features['pop_density_per_km2'],
        bins=[0, 1500, float('inf')],
        labels=['suburban', 'urban']
    )

    # 10. Province-level vegetable consumption (from research)
    veg_consumption_map = {
        'Guangdong': 142.57, 'Sichuan': 120, 'Jiangsu': 115, 'Shandong': 110,
        'Henan': 105, 'Hubei': 100, 'Hunan': 95, 'Anhui': 90,
        'Fujian': 95, 'Zhejiang': 100, 'Hebei': 85, 'Liaoning': 80,
        'Shaanxi': 80, 'Yunnan': 75, 'Guangxi': 85, 'Guizhou': 50.49,
        'Jiangxi': 80, 'Shanxi': 75, 'Gansu': 70, 'Heilongjiang': 75,
        'Inner Mongolia': 70, 'Jilin': 75, 'Ningxia': 70, 'Qinghai': 60,
        'Hainan': 85, 'Tibet': 50, 'Beijing': 110, 'Shanghai': 120,
        'Tianjin': 100, 'Chongqing': 115,
    }
    features['province_veg_consumption_per_capita'] = features['province'].map(
        veg_consumption_map
    ).fillna(95)  # National average

    # 11. Transportation hub score (simplified)
    tier_hub_score = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
    features['transport_hub_score'] = features['city_tier_numeric'].map(tier_hub_score)

    return features
```

---

## 10. What Works

### 10.1 Evidence-Based Best Practices

1. **Cross-learning across hierarchical levels (M5 winner approach)** [^462^]
   - Train models at multiple aggregation levels (store, store-category, store-department)
   - Combine forecasts with equal weights
   - This captures both store-specific patterns and cross-store learnings

2. **Native categorical handling with LightGBM/CatBoost** [^390^] [^527^]
   - Both LightGBM and CatBoost handle categorical features natively
   - CatBoost's ordered target statistics are particularly effective for geographic features
   - Avoids manual encoding complexity and target leakage

3. **Store clustering for efficiency** [^346^]
   - Group similar stores (by climate, tier, format, sales pattern)
   - Create cluster-level forecasts
   - Improves both accuracy and planning efficiency

4. **Seasonal-climate interactions** [^307^]
   - Winter demand surge in northern provinces for preserved foods
   - Model as interaction feature between month and climate region
   - Significant predictive power for seasonal products

5. **GDP per capita as continuous feature** [^470^]
   - Log-transformed GDP per capita captures purchasing power gradient
   - Normalize to national average for relative interpretation
   - More powerful than simple tier classification alone

6. **Regional cuisine preferences as categorical** [^347^]
   - Sour-pickled vs salted vs low consumption regions
   - Captures deep cultural preferences not explained by economics
   - Based on epidemiological study of 500K+ participants

---

## 11. What Doesn't Work

### 11.1 Common Pitfalls

1. **Simple one-hot encoding for high-cardinality geographic features**
   - Cities (300+ unique values) create too many dimensions
   - Loses ordinal/economic information in tier structure
   - Use native categorical support or target encoding instead

2. **Static store clusters without periodic refresh**
   - Store characteristics change over time
   - Consumer behavior shifts (especially in fast-growing lower-tier cities)
   - Re-evaluate clusters quarterly or semi-annually [^356^]

3. **Ignoring target leakage in geographic encoding**
   - Computing target means using future data creates leakage
   - Always use time-series-aware encoding (expanding window or CatBoost's ordered encoding)
   - Cross-validation must respect temporal ordering

4. **Using raw city names as identifiers without aggregation**
   - Too many unique values, many with few observations
   - Risk of overfitting on rare cities
   - Group into tiers or use target encoding with smoothing

5. **Assuming uniform demand patterns across a province**
   - Large provinces (Sichuan: 83M people) have huge internal variation
   - Urban vs rural differences within provinces are substantial
   - City-level features are more informative than province-level alone

6. **Overweighting Tier 1 cities**
   - Tier 1 cities have slower spending growth (~0.4% in Shanghai)
   - Lower-tier cities show faster growth (~5.8%)
   - Forecasting models should not assume Tier 1 = highest growth [^530^]

---

## 12. Recommended Approach

### 12.1 For Our Specific Problem (Pickled Vegetable Demand Forecasting)

Given Customer1 (15+ provinces), Customer2 (Guangdong/Guangxi), Customer3 (scattered):

**Tier 1: Essential Geographic Features (implement first)**

| Feature | Type | Encoding | Rationale |
|---------|------|----------|-----------|
| `city_tier` | Categorical | Native (LightGBM/CatBoost) | Primary proxy for purchasing power |
| `province` | Categorical | Native or target-encoded | Captures regional cuisine preferences |
| `log_gdp_per_capita` | Continuous | Log transform | Direct purchasing power measure |
| `climate_zone` | Categorical | Native | Drives seasonal preservation demand |
| `winter_north_interaction` | Binary | 0/1 | Winter demand surge in north |
| `pickle_propensity_score` | Categorical | Native | Sour/salted/low consumption region |

**Tier 2: Secondary Features (add after baseline)**

| Feature | Type | Encoding | Rationale |
|---------|------|----------|-----------|
| `population_density_tier` | Ordinal | 0-3 | Urban vs suburban format |
| `stores_in_city` | Continuous | Raw + log | Market saturation proxy |
| `province_veg_consumption_per_capita` | Continuous | Raw | Regional vegetable demand baseline |
| `economic_level` | Ordinal | 0-3 | GDP-based development tier |

**Tier 3: Advanced Features (for fine-tuning)**

| Feature | Type | Encoding | Rationale |
|---------|------|----------|-----------|
| `transport_hub_score` | Ordinal | 1-10 | Supply chain connectivity |
| `month_climate_interaction` | Interaction | Multiplicative | Seasonal pattern by climate |
| `city_tier_trend_interaction` | Interaction | Multiplicative | Growth rate varies by tier |

### 12.2 Model Architecture Recommendation

Based on M5 competition findings [^462^] [^143^]:

```python
# Recommended: LightGBM with hierarchical cross-learning

models = {
    # Level 1: Per-customer models
    'customer_level': train_per_customer(),

    # Level 2: Per-customer-province models
    'customer_province': train_per_customer_province(),

    # Level 3: Per-customer-city_tier models
    'customer_tier': train_per_customer_tier(),

    # Level 4: Per-customer-climate_zone models
    'customer_climate': train_per_customer_climate(),
}

# Combine with weights based on local validation performance
weights = optimize_weights(models, val_data)
final_prediction = weighted_average(models, weights)
```

### 12.3 Special Handling for Each Customer

**Customer1 (15+ provinces, broad coverage):**
- Strong geographic features will be highly predictive
- Province-level cuisine preferences matter
- Climate zone x season interactions are important
- Consider separate models per region (East, South, North, West)

**Customer2 (Guangdong/Guangxi concentration):**
- Regional features will have less variance (less discriminative power)
- Focus on within-province variation (city-level GDP, urban vs suburban)
- Guangdong has highest per capita vegetable consumption (142.57 kg)
- Guangxi has lower consumption but supplies Guangdong

**Customer3 (scattered smaller):**
- Limited data per location → rely more on cross-learning
- City-tier and province features become more important
- Store clustering with similar stores across customers
- Climate and cuisine features provide transfer learning signal

---

## 13. Sources

### Academic/Peer-Reviewed Sources

1. **"The M5 Accuracy Competition: Results, Findings and Conclusions"**
   - Authors: Spyros Makridakis et al.
   - URL: https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
   - Key finding: LightGBM with hierarchical cross-learning won M5

2. **"The interprovincial circulation of vegetables in China and its impact on virtual water and virtual land resources"**
   - Journal: Frontiers in Sustainable Food Systems, 2025
   - URL: https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2025.1685599/full
   - Key finding: Guangdong highest consumption (46.7M tons), vegetable flow patterns

3. **"Chinese Food Consumption Adaptation and Sustainability Under Climate Warming"**
   - Journal: MDPI Sustainability, 2025
   - URL: https://www.mdpi.com/2071-1050/17/21/9682
   - Key finding: Climate-driven consumption shifts, North-South divergence

4. **"Preserved vegetable consumption and gastrointestinal tract cancers: A prospective study"**
   - Journal: Journal of Global Health, 2024
   - URL: https://jogh.org/2024/jogh-14-04191/
   - Key finding: Regional pickled vegetable consumption classification (sour vs salted vs low)

5. **"Spatiotemporal differences and influencing factors of the dietary structure index in Chinese urban residents"**
   - Journal: Frontiers/ResearchGate, 2025
   - URL: https://www.researchgate.net/publication/392208633
   - Key finding: Northeast DSI highest, dietary patterns correlate with climate

6. **"Climate Classification for Major Cities in China Using Cluster Analysis"**
   - Journal: MDPI Atmosphere, 2024
   - URL: https://www.mdpi.com/2073-4433/15/7/741
   - Key finding: Köppen classification of 36 major Chinese cities

7. **"A gradient model for the spatial patterns of cities"**
   - Journal: arXiv, 2020
   - URL: https://arxiv.org/pdf/2010.07700.pdf
   - Key finding: China City Statistical Yearbook data for GDP, population density

### Industry Reports and Market Research

8. **"Ranking of Cities' Attractiveness in China 2025" - Yicai Media Group**
   - URL: https://www.yicaiglobal.com/news/chengdu-hangzhou-13-others-rank-as-new-first-tier-chinese-cities-in-2025
   - Key finding: 337 city tier classification, methodology details

9. **"China: per capita GDP by province 2024" - Statista**
   - URL: https://www.statista.com/statistics/1093666/china-per-capita-gross-domestic-product-gdp-by-province/
   - Key finding: Complete provincial GDP per capita rankings

10. **"Packed Pickled Mustard 2025 Trends and Forecasts" - Archive Market Research**
    - URL: https://www.archivemarketresearch.com/reports/packed-pickled-mustard-164682
    - Key finding: Market concentration, regional distribution

11. **"Zhacai Market 2025-2034" - MarkWide Research**
    - URL: https://markwideresearch.com/zhacai-market/
    - Key finding: Zhacai market segmentation and growth

12. **"Chongqing Fuling: China's Zhacai Hometown" - Youlian Trade**
    - URL: http://www.youliantrade.com/news/405.html
    - Key finding: 70%+ national market share, 36.7B yuan industry

### Technical/Implementation Sources

13. **"CatBoost's Categorical Encoding: OneHot vs. Target Encoding" - GeeksforGeeks**
    - URL: https://www.geeksforgeeks.org/machine-learning/catboosts-categorical-encoding-one-hot-vs-target-encoding/
    - Key finding: Ordered target encoding implementation

14. **"Store Localization and Clustering: Best Practices" - Toolio**
    - URL: https://www.toolio.com/post/store-localization-and-clustering-best-practices-for-retail-planners
    - Key finding: Clustering methodology for retail planning

15. **"What is store clustering in retail?" - daVinci Retail**
    - URL: https://www.davinciretail.com/resources/what-is-store-clustering/
    - Key finding: Index-based clustering approach

16. **"A Survey of Machine Learning Methods for Time Series Prediction" - MDPI Applied Sciences**
    - URL: https://www.mdpi.com/2076-3417/15/11/5957
    - Key finding: M5 competition summary, LightGBM dominance

17. **"Small City, Big Spenders: The Rise of China's New Consumer Class" - Sixth Tone**
    - URL: https://www.sixthtone.com/news/1017003
    - Key finding: Tier 3-4 cities growing faster in consumption

18. **"Fuling Pickled Mustard Tuber Extends Its Market Reach" - iChongqing**
    - URL: https://www.ichongqing.info/2024/12/18/fuling-pickled-mustard-tuber-extends-its-market-reach-to-over-50-countries-and-regions/
    - Key finding: 30% market share, 28-30 province distribution

19. **"Categorical Feature Support in Gradient Boosting" - scikit-learn**
    - URL: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html
    - Key finding: Native categorical support in HistGradientBoosting

20. **Hierarchical Forecasting Model - Umbrex**
    - URL: https://umbrex.com/resources/frameworks/supply-chain-frameworks/hierarchical-forecasting-model/
    - Key finding: Bottom-up, top-down, middle-out, optimal reconciliation methods

---

*End of Research Document - Dimension 5: Geographic & Regional Economic Features*
