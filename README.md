# HKU AI Competition Problem 2 - Award-Winning Solution

## Multi-Model Ensemble with External Data Integration

### Models
- **LightGBM** (Tweedie loss) — nonlinear pattern capture
- **SBA** (Syntetos-Boylan Approximation) — intermittent demand specialist
- **Ridge Regression** — trend extrapolation

### Features (40+)
- Chinese holidays (auto-detected for any year via python-holidays)
- 24 Solar Terms (节气)
- Day-of-week cyclic encodings
- Lag features (1, 2, 7, 14, 21 days)
- Rolling statistics (7, 14, 21 days)
- Mean encodings (customer, category, customer×category, province, DOW)
- Optional: Weather data (Open-Meteo API)

### Key Innovation: ADIDA + DOW Distribution
1. Predict monthly total per customer×category from historical average
2. Distribute to daily using learned day-of-week proportions
3. Ensures realistic, non-zero predictions with natural weekly rhythm

### Generic Support
- **Any year**: Auto-detects year from data, generates correct holidays
- **Any data format**: Auto schema detection finds columns by pattern matching
- **Robust**: Handles different column names, missing sheets, different date ranges

### Usage
```bash
pip install -r requirements.txt
python main.py --train-data "jan.xlsx" "feb_mar.xlsx"
```
