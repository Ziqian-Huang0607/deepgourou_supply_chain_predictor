# Problem 2: File Analysis — Demand Forecasting for Supply Chain

## File Inventory

| File | Type | Content Summary |
|------|------|-----------------|
| `绝配-港大AI赛题-上海中学_题目常见问题补充.docx` | Problem Statement + FAQ | Complete problem description, scoring criteria, Q&A for both Problem 1 (ChatBI) and Problem 2 (Forecasting) |
| `题目一QA测试和案例.docx` | QA Test Cases + Answers | Example questions and detailed answers for Problem 1, showing data structure and query logic |
| `测试客户下单2-3月V2.xlsx` | Data (Feb-Mar) | Order data for February-March 2026 period |
| `测试客户下单量1月V2.xlsx` | Data (January) | Order data for January 2026 period |

## Key Problem Understanding (from Problem Statement)

### Problem 2: Open-ended Outcome-based Forecasting Model
- **Data provided**: Order Basic Info Table + Customer Order Details Table (two full tables)
- **Prediction target**: Each customer's future demand for different products (quantity, category, timing)
- **Time horizon**: Predict Month 3 (April 2026) based on 2 months of history (Jan + Feb-Mar)
- **Critical insight from organizers**: "Purely based on time series data within orders will give poor results" (单纯基于订单内的时序数据进行预测将会得到不太好的结果)
- **External data explicitly encouraged**: Historical traffic data, field research data, weather, economic indicators, etc.

### Scoring Criteria
- **No restrictions** on algorithms: Transformer, LSTM, RNN, logistic regression, pure theoretical analysis all acceptable
- **Scoring based on**: Match between predicted Month 3 values and actual business data
- **Output format**: Structured, queryable, comparable numerical data
- **Prediction scope**: Time = known data + 1 month. Content = per customer, per product category quantity and timing

### Industry Context
- **Company**: 上海绝配柔性供应链科技有限公司 (Shanghai Juepei Flexible Supply Chain Technology)
- **Partners**: Hong Kong University + Shanghai High School
- **Data characteristics**: Supply chain/logistics data with multiple customers, warehouses, stores across China
- **Product types**: Food products (脆卜装/pickled vegetables, etc. — visible in data)

## Per-File Extraction

### File 1: Problem Statement
**Key Claims:**
1. Pure time-series approaches on internal data alone are insufficient
2. External data collection is not just allowed but expected for good results
3. Any algorithm is acceptable — innovation in approach is valued
4. Self-testing through train/test split is encouraged

**Scoring Strategy Implications:**
- Organizers WANT to see creative external data integration
- This is a signal that the best solutions will go beyond standard ML forecasting
- "Open-ended" means creative/ interdisciplinary approaches are rewarded

### File 2: QA Test Cases
**Key Data Structure Insights:**
- Three main tables: Order Basic Info, Customer Order Details, Order Operation Details
- Key fields: Order ID (订单单号), Customer Code (货主编码), Product Code (商品编码), Quantity (求和项:预计发货数量EA), Creation Time (创建时间), Warehouse (仓库), Store (收货门店), Province-City-District (省市区)
- Operations tracked: Order review status, processing timestamps

**Query Patterns Observed:**
- Date range filtering on creation time
- Customer aggregation and comparison
- Product category grouping
- Deduplication (distinct order counts vs. line items)
- Geographical analysis (province-city-district hierarchy)

### File 3 & 4: Order Data (Jan + Feb-Mar)
**Data Structure:**
- Columns: 订单单号, 订单类型, 货主编码, 货主, 仓库, 收货门店, 省市区, 求和项:预计发货数量EA, 预计总箱数, 创建人, 创建时间
- Order types: 销售出库 (sales outbound), 其他出库 (other outbound)
- 3 customers: C01 (客户1), C02 (客户2), C03 (客户3/钱小匠)
- Multiple warehouses across China: 深圳5仓, 武汉1仓, 广州6仓, 合肥1仓, 杭州1仓, 成都2仓, etc.
- Geographical coverage: Guangdong, Hubei, Zhejiang, Jiangsu, Anhui, Sichuan, Shaanxi, etc.

**Initial Pattern Observations:**
- Customer1 has highest volume, most diverse store network
- Customer2 has concentrated stores in Guangdong province
- Customer3 (钱小匠) has smaller orders, different store naming pattern
- Daily ordering patterns visible
- Different warehouse regions serve different geographical areas

## Cross-File Mapping

| Theme | File Sources | Complementarity |
|-------|-------------|-----------------|
| Data Structure | File 1 (schema), File 2 (examples), File 3&4 (raw data) | Complete picture of all available fields |
| Scoring Strategy | File 1 (explicit criteria) | External data integration is KEY differentiator |
| Query Patterns | File 2 (QA examples) | Shows what kinds of analysis are expected |
| Industry Context | File 1 (company name), File 3&4 (product names) | Food/retail supply chain in China |

## Gap Analysis — Critical for Winning

1. **No external data provided** — This is BY DESIGN. The organizers expect teams to FIND and INTEGRATE external data. This is the main differentiation opportunity.

2. **No explicit product category mapping** — Need to derive product categories from product names/codes

3. **No store-level features** — Need to engineer features from store names, locations

4. **No temporal features** — Chinese holidays, weekends, weather patterns not provided

5. **No economic/regional context** — GDP, population, local economic activity data not provided

6. **No competitive/market context** — Industry trends, seasonal food consumption patterns not provided

## Consolidated Theme List for Research

1. **SOTA Time Series Forecasting** — DeepAR, N-BEATS, N-HiTS, TFT, PatchTST for demand forecasting
2. **Supply Chain External Data Integration** — Weather, traffic, holidays, events, economic indicators
3. **Chinese Food Industry Seasonality** — Lunar New Year effects, regional cuisine patterns, seasonal food demand
4. **Kaggle Competition Winning Strategies** — Demand forecasting competitions, feature engineering secrets
5. **Geographic Demand Modeling** — Regional economic differences, urban vs rural consumption
6. **Multi-modal Data Fusion** — Combining tabular + external API data for prediction
7. **Uncertainty Quantification** — Probabilistic forecasting, prediction intervals (professors love this)
8. **Causal Inference for Demand** — CausalML, DoWhy, understanding WHY demand changes
9. **Automated Feature Engineering** — TSFresh, Featuretools for time series
10. **Neural Architecture Search** — AutoML for forecasting, finding optimal model structures
