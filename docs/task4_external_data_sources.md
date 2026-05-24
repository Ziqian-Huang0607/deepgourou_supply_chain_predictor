# Free External Data Sources for Chinese Cities - Demand Forecasting

> **Research Date**: 2025  
> **Purpose**: Identify FREE external data APIs and sources that can improve demand forecasting models for Chinese cities  
> **Coverage**: Weather, holidays, economic indicators, traffic, air quality, city classification, and calendar utilities

---

## Table of Contents

1. [Weather Data APIs](#1-weather-data-apis)
   - [1.1 Open-Meteo (RECOMMENDED - Best Free Option)](#11-open-meteo)
   - [1.2 OpenWeatherMap](#12-openweathermap)
   - [1.3 Visual Crossing Weather API](#13-visual-crossing-weather-api)
   - [1.4 Meteostat (Python Library)](#14-meteostat)
   - [1.5 China Meteorological Administration (CMA) Data](#15-china-meteorological-administration)
2. [Holiday & Calendar Data](#2-holiday--calendar-data)
   - [2.1 Jiejiari API (Chinese Holiday API)](#21-jiejiari-api)
   - [2.2 Nager.Date API](#22-nagerdate-api)
   - [2.3 Chinese Holidays Python Library (lunarcalendar)](#23-chinese-holidays-python-library)
3. [Economic & Demographic Data](#3-economic--demographic-data)
   - [3.1 China Data Portal (chinadata.live)](#31-china-data-portal)
   - [3.2 World Bank Open Data API](#32-world-bank-open-data)
   - [3.3 National Bureau of Statistics of China](#33-national-bureau-of-statistics)
4. [Air Quality Data](#4-air-quality-data)
   - [4.1 WAQI (World Air Quality Index) API](#41-waqi-api)
5. [Traffic & Transportation Data](#5-traffic--transportation-data)
   - [5.1 Gaode (AMap) API](#51-gaode-amap-api)
   - [5.2 Baidu Maps API](#52-baidu-maps-api)
   - [5.3 PTV Flows API](#53-ptv-flows-api)
6. [City Classification Data](#6-city-classification-data)
   - [6.1 Yicai Global City Tier Rankings](#61-yicai-global-city-tier-rankings)
   - [6.2 GYBrand China Top 100 Cities Index](#62-gybrand-china-top-100-cities)
7. [Summary Comparison Table](#7-summary-comparison-table)
8. [Implementation Recommendations](#8-implementation-recommendations)

---

## 1. Weather Data APIs

### 1.1 Open-Meteo (RECOMMENDED - Best Free Option)

**The #1 choice for free weather data for Chinese cities.** No API key required, completely free for non-commercial use.

| Attribute | Details |
|-----------|---------|
| **URL** | https://open-meteo.com |
| **API Docs** | https://open-meteo.com/en/docs |
| **Free Tier** | Unlimited requests (fair use policy for non-commercial) |
| **API Key Required** | No - completely open access |
| **Rate Limit** | No hard limits; ~10,000 requests/day recommended for non-commercial |
| **Historical Data** | Back to 1940 |
| **Forecast** | Up to 16 days |
| **Weather Variables** | 80+ (temperature, humidity, precipitation, wind, solar, UV, soil, etc.) |
| **Air Quality** | Yes - separate API (PM2.5, PM10, CO, NO2, O3) |
| **Open Source** | Yes (AGPLv3) - can self-host |

**Coverage for China:**
- Uses CMA (China Meteorological Administration) data model at `/v1/cma` endpoint
- Global coverage via lat/lon - works for any Chinese city
- Supports geocoding API to find coordinates by city name
- Timezone auto-detection for all Chinese cities

**Key API Endpoints:**
```
# Weather forecast
https://api.open-meteo.com/v1/forecast?latitude=31.23&longitude=121.47&current_weather=true

# CMA China model specifically
https://api.open-meteo.com/v1/cma?latitude=39.90&longitude=116.41&hourly=temperature_2m

# Historical weather data
https://archive-api.open-meteo.com/v1/archive?latitude=39.90&longitude=116.41&start_date=2020-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum

# Geocoding (find coordinates by city name)
https://geocoding-api.open-meteo.com/v1/search?name=Beijing&count=1

# Air quality
https://air-quality-api.open-meteo.com/v1/air-quality?latitude=31.23&longitude=121.47&current=pm10,pm2_5
```

**Python Example:**
```python
import requests

def get_weather_china(city_lat, city_lon, start_date, end_date):
    """Fetch daily weather for any Chinese city"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                 "precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max,"
                 "shortwave_radiation_sum,relative_humidity_2m_mean",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    return response.json()

# Beijing: 39.9042, 116.4074
# Shanghai: 31.2304, 121.4737
# Shenzhen: 22.5431, 114.0579
data = get_weather_china(31.2304, 121.4737, "2023-01-01", "2024-12-31")
```

**Best For:** Historical weather features, temperature trends, precipitation, seasonal patterns

---

### 1.2 OpenWeatherMap

| Attribute | Details |
|-----------|---------|
| **URL** | https://openweathermap.org |
| **Free Tier** | 60 calls/minute (1,000 calls/day), requires registration |
| **API Key Required** | Yes - free signup |
| **Historical Data** | Available via One Call API 3.0 (1,000 free calls/day) |
| **Historical Archive** | 47+ years back (One Call API 3.0) |
| **Coverage** | 200,000+ cities worldwide including all major Chinese cities |

**Key Products:**
- **Current Weather API** - Free tier
- **One Call API 3.0** - 1,000 free calls/day, includes historical, forecast, alerts
- **Air Pollution API** - Current, forecast, historical air quality data

**Python Example:**
```python
import requests

API_KEY = "your_free_api_key"

# Current weather for Shanghai
url = f"https://api.openweathermap.org/data/2.5/weather?q=Shanghai,CN&appid={API_KEY}&units=metric"
response = requests.get(url)
data = response.json()
```

**Limitations:**
- Free tier is limited; historical data requires One Call API 3.0 subscription
- Requires API key management
- Rate limits may constrain bulk data collection

**Best For:** Current conditions, short-term forecasts, air pollution data

---

### 1.3 Visual Crossing Weather API

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.visualcrossing.com/weather-api |
| **Free Tier** | 1,000 records/day (includes commercial use!) |
| **API Key Required** | Yes - free signup |
| **Historical Data** | 50+ years back (to ~1975) |
| **Forecast** | 15-day forecast |
| **Pricing** | $0.0001/record for metered plan beyond free tier |

**Key Features:**
- Single API call for historical + forecast + current conditions
- Supports address, ZIP code, city name, or lat/lon
- Nearly 100 weather elements available
- Query Builder and AI Code Generator for easy integration
- CSV, JSON, or FlatJSON output formats

**Python Example:**
```python
import requests

API_KEY = "your_api_key"
LOCATION = "Shanghai,CN"
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}/{START_DATE}/{END_DATE}?unitGroup=metric&key={API_KEY}&include=days"
response = requests.get(url)
data = response.json()
```

**Best For:** Bulk historical data download, commercial projects on free tier

---

### 1.4 Meteostat (Python Library)

| Attribute | Details |
|-----------|---------|
| **URL** | https://meteostat.net |
| **Python Library** | `pip install meteostat` |
| **Free Tier** | Unlimited access, no signup required |
| **API Key Required** | No (Python library uses bulk data interface) |
| **JSON API** | Available (free for non-commercial use) |
| **Data Sources** | NOAA, DWD, and other national weather services |
| **Coverage** | Global weather station data |

**Key Features:**
- Works with Pandas DataFrames directly
- Point-based interpolation for locations without weather stations
- Hourly, daily, and monthly data available
- Weather station search by proximity

**Python Example:**
```python
from meteostat import Point, Daily
import pandas as pd
from datetime import datetime

# Define location (Beijing)
beijing = Point(39.9042, 116.4074, 44)

# Define time range
start = datetime(2023, 1, 1)
end = datetime(2024, 12, 31)

# Get daily data
data = Daily(beijing, start, end)
df = data.fetch()

# df contains: tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun
print(df.head())
```

**Note:** Meteostat uses weather station data. Coverage in China depends on proximity to meteorological stations. Major cities are well covered.

**Best For:** Python-based workflows, Pandas integration, long time series analysis

---

### 1.5 China Meteorological Administration (CMA) Data

| Attribute | Details |
|-----------|---------|
| **URL** | http://data.cma.cn/ |
| **Type** | Official government weather data |
| **Free Tier** | Requires registration; some data free, some requires payment |
| **Language** | Chinese |
| **Data Quality** | Highest quality - official meteorological observations |

**Limitations:**
- Website and documentation in Chinese only
- May require approval for data access
- Some datasets require payment
- More bureaucratic than international alternatives

**Best For:** When highest data accuracy is required; validation against other sources

---

## 2. Holiday & Calendar Data

### 2.1 Jiejiari API (Chinese Holiday API)

**The best free API for Chinese holidays specifically designed for Chinese calendar.**

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.jiejiariapi.com/en |
| **Free Tier** | 50 requests/day (anonymous) / 100 requests/day (with free API key) |
| **API Key Required** | Optional - higher limits with key |
| **Commercial Use** | Not allowed on free tier (personal testing only) |

**API Endpoints:**
```
# Get full year holidays
GET https://api.jiejiariapi.com/v1/holidays/2026

# Get all weekends for a year
GET https://api.jiejiariapi.com/v1/weekends/2026

# Check if a specific date is a workday
GET https://api.jiejiariapi.com/v1/workday/2026/02/17
```

**Response includes:**
- Holiday name (Chinese and English)
- Holiday type (national holiday, working day, weekend)
- "Tiaoxiu" (调休) - swap rest days info
- Whether a day is a working day or rest day

**Python Example:**
```python
import requests

# Get all Chinese holidays for 2026
url = "https://api.jiejiariapi.com/v1/holidays/2026"
response = requests.get(url)
holidays = response.json()

for holiday in holidays:
    print(f"{holiday['date']}: {holiday['name']} - {'Holiday' if holiday['is_off_day'] else 'Workday'}")
```

**Paid Plans:**
- Basic: 59 CNY/year - 10,000 requests/day
- Pro: 99 CNY/year - 30,000 requests/day
- Business: 1,200 CNY/year - unlimited

**Best For:** Identifying Chinese public holidays, "tiaoxiu" (调休) swap days, workday vs. rest day classification

---

### 2.2 Nager.Date API

**Simple, free API for public holidays across all countries including China.**

| Attribute | Details |
|-----------|---------|
| **URL** | https://date.nager.at |
| **Free Tier** | Unlimited (no API key needed for basic usage) |
| **API Key Required** | No |
| **Rate Limit** | None specified |
| **Coverage** | 100+ countries |

**API Endpoints:**
```
# Get China's public holidays for a year
GET https://date.nager.at/api/v3/publicholidays/2026/CN

# Get available countries
GET https://date.nager.at/api/v3/availablecountries

# Check if today is a holiday
GET https://date.nager.at/api/v3/istodaypublicholiday/CN
```

**Response Fields:**
- `date` - Holiday date
- `localName` - Name in local language (Chinese)
- `name` - English name
- `countryCode` - CN
- `fixed` - Whether date is fixed
- `global` - Whether national holiday
- `launchYear` - Year the holiday was established
- `types` - Public, Bank, School, etc.

**Python Example:**
```python
import requests

url = "https://date.nager.at/api/v3/publicholidays/2026/CN"
response = requests.get(url)
holidays = response.json()

for h in holidays:
    print(f"{h['date']}: {h['localName']} ({h['name']})")
```

**2026 China Holidays via Nager.Date:**
| Date | Holiday |
|------|---------|
| Jan 1 | New Year's Day (元旦) |
| Feb 17-23 | Spring Festival / Chinese New Year (春节) - 9 days |
| Apr 4-6 | Qingming / Tomb Sweeping (清明节) |
| May 1-5 | Labour Day (劳动节) |
| Jun 19-21 | Dragon Boat Festival (端午节) |
| Sep 25-27 | Mid-Autumn Festival (中秋节) |
| Oct 1-7 | National Day / Golden Week (国庆节) |

**Limitations:**
- Does NOT include "tiaoxiu" (调休) swap days - only statutory holidays
- Does not distinguish working weekends vs. rest weekends
- Less granular than Jiejiari API

**Best For:** Quick integration, cross-country holiday comparison, basic holiday flagging

---

### 2.3 Chinese Holidays Python Library (lunarcalendar)

**For lunar calendar conversion and traditional Chinese date features.**

| Attribute | Details |
|-----------|---------|
| **Library** | `lunar_python` (recommended) |
| **Installation** | `pip install lunar_python` |
| **Alternative** | `pip install lunarcalendar` |
| **Cost** | Free, open source |
| **No API Required** | Works offline |

**lunar_python Features:**
- Solar (Gregorian) to Lunar calendar conversion
- Chinese zodiac (生肖)
- Gan-Zhi (干支) - heavenly stems and earthly branches
- Solar terms (二十四节气)
- Buddhist and Taoist calendars
- No third-party dependencies

**Python Example:**
```python
from lunar_python import Solar, Lunar

# Create a solar date
solar = Solar.fromYmd(2025, 1, 29)

# Convert to lunar
lunar = solar.getLunar()
print(f"Lunar date: {lunar.toFullString()}")

# Chinese zodiac
print(f"Zodiac: {lunar.getYearShengXiao()}")

# Gan-Zhi (干支)
print(f"Year Gan-Zhi: {lunar.getYearInGanZhi()}")

# Solar term (节气)
print(f"Current solar term: {lunar.getJieQi()}")
```

**lunarcalendar Library (Alternative):**
```python
from lunarcalendar import Converter, Solar, Lunar

# Solar to Lunar
lunar = Converter.Solar2Lunar(Solar(2025, 1, 29))
print(f"Lunar: {lunar.year}-{lunar.month}-{lunar.day}")
```

**Demand Forecasting Feature Engineering:**
```python
def add_chinese_calendar_features(df, date_col):
    """Add Chinese calendar features to a DataFrame"""
    from lunar_python import Solar, Lunar
    
    def extract_features(date):
        solar = Solar.fromDate(date)
        lunar = solar.getLunar()
        return pd.Series({
            'lunar_month': lunar.getMonth(),
            'lunar_day': lunar.getDay(),
            'is_lunar_month_1': lunar.getMonth() == 1,  # Spring Festival month
            'is_lunar_month_8': lunar.getMonth() == 8,  # Mid-Autumn month
            'zodiac': lunar.getYearShengXiao(),
            'is_solar_term': lunar.getJieQi() is not None,
        })
    
    features = df[date_col].apply(extract_features)
    return pd.concat([df, features], axis=1)
```

**Best For:** Feature engineering - lunar month/day, solar terms, zodiac year; cultural/seasonal pattern detection

---

## 3. Economic & Demographic Data

### 3.1 China Data Portal (chinadata.live)

**The best FREE English-language source for China's official statistics with REST API.**

| Attribute | Details |
|-----------|---------|
| **URL** | https://chinadata.live |
| **API Base** | `https://chinadata.live/api/v2` |
| **Free Tier** | Completely free, no API key required |
| **Rate Limit** | Recommend <100 requests/minute |
| **Data Format** | JSON and CSV |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Commercial Use** | Yes, with attribution |

**Available Datasets (99+ datasets):**

| Category | Datasets |
|----------|----------|
| **Economy** | GDP, GDP per capita, inflation (CPI), PPI, M2 money supply, foreign reserves |
| **Trade** | Monthly customs trade data with 40+ countries, bilateral trade flows |
| **Population** | Total population, births, deaths, urbanization rate |
| **Energy** | Coal, solar, wind, oil production and consumption |
| **Manufacturing** | Steel, auto, electronics production |
| **Agriculture** | Crop production, livestock |
| **Infrastructure** | Rail, roads, ports |
| **Society** | Education, health, tourism |

**Key API Endpoints:**
```python
# List all datasets
GET https://chinadata.live/api/v2/datasets

# Search datasets
GET https://chinadata.live/api/v2/search?q=population

# Get specific dataset
GET https://chinadata.live/api/v2/data/gdp
GET https://chinadata.live/api/v2/data/china-gdp-per-capita
GET https://chinadata.live/api/v2/data/china-total-population
GET https://chinadata.live/api/v2/data/china-births
GET https://chinadata.live/api/v2/data/china-inbound-tourism
```

**Python Example:**
```python
import requests
import pandas as pd

# Get China GDP data
response = requests.get('https://chinadata.live/api/v2/data/gdp')
dataset = response.json()['data']

df = pd.DataFrame(dataset['data'])
df['date'] = pd.to_numeric(df['date'])
df['value'] = pd.to_numeric(df['value'])

print(f"Dataset: {dataset['title']}")
print(f"Unit: {dataset['unit']}")
print(f"Source: {dataset['source']}")
print(df.tail(10))
```

**Sample Data Available:**
- China GDP: 1949-2025 (77 data points)
- GDP per capita: 1949-2025 (CNY 119 to CNY 99,665)
- Total population: 2010-2025
- Annual births: 1949-2025
- Inbound tourism: 1978-2025

**Trade Data API (Special Feature):**
```python
# China-US trade data (monthly, by HS chapter)
GET https://chinadata.live/api/v2/trade/country/united-states

# Response includes:
# - monthly exports/imports for 92 months
# - export breakdown by HS commodity chapter
# - latest month data
```

**Best For:** National-level economic trends, trade flows, population trends, inflation indicators

---

### 3.2 World Bank Open Data

| Attribute | Details |
|-----------|---------|
| **URL** | https://data.worldbank.org |
| **API Base** | `https://api.worldbank.org/v2` |
| **Free Tier** | Completely free, no API key required |
| **Rate Limit** | None for normal usage |
| **Indicators** | 16,000+ indicators, 200+ countries |
| **Historical Data** | From 1960 |
| **China Indicators** | 1,400+ indicators |

**Key China Indicators:**
| Indicator Code | Description |
|---------------|-------------|
| `NY.GDP.MKTP.CD` | GDP (current US$) |
| `NY.GDP.PCAP.CD` | GDP per capita (current US$) |
| `SP.POP.TOTL` | Total population |
| `FP.CPI.TOTL` | Consumer Price Index |
| `SL.UEM.TOTL.ZS` | Unemployment rate |
| `EN.ATM.CO2E.KT` | CO2 emissions |
| `SE.XPD.TOTL.GD.ZS` | Education expenditure (% of GDP) |

**Python Example:**
```python
import requests

# Get China GDP
url = "https://api.worldbank.org/v2/country/CN/indicator/NY.GDP.MKTP.CD"
params = {"format": "json", "per_page": 100, "date": "2010:2024"}
response = requests.get(url, params=params)
data = response.json()

# data[1] contains the actual data points
for record in data[1]:
    print(f"{record['date']}: {record['value']}")
```

**R Library (ChinAPIs):**
```r
# Install ChinAPIs package
install.packages("ChinAPIs")
library(ChinAPIs)

# Get China CPI
cpi <- get_china_cpi()

# Get China holidays
holidays <- get_china_holidays(2026)

# Get China GDP
gdp <- get_china_gdp()

# Get China population
pop <- get_china_population()
```

**Best For:** Cross-country comparisons, long-term economic trends, standardized development indicators

---

### 3.3 National Bureau of Statistics of China

| Attribute | Details |
|-----------|---------|
| **URL (English)** | https://www.stats.gov.cn/english/ |
| **URL (Chinese)** | https://www.stats.gov.cn/ |
| **Data** | Official statistics, press releases, annual data |
| **Cost** | Free |
| **API** | No API; data available via web and downloads |

**Available Data:**
- National GDP, GDP growth by quarter
- Population statistics (urban/rural, age structure)
- Employment and wages
- Consumer prices (CPI)
- Industrial production
- Fixed asset investment
- Retail sales
- Foreign trade
- Fiscal revenue and expenditure

**Limitations:**
- No direct API
- Data often in press release format
- City-level data may be scattered
- English version may lag behind Chinese version

**Best For:** Validation of other data sources, official government figures

---

## 4. Air Quality Data

### 4.1 WAQI (World Air Quality Index) API

| Attribute | Details |
|-----------|---------|
| **URL** | https://aqicn.org/api/ |
| **Free Tier** | Yes - requires API token (free registration) |
| **Rate Limit** | 1,000 requests/second (very generous) |
| **Coverage** | 11,000+ station-level, 1,000+ city-level |

**China Coverage:**
- Major cities: Beijing, Shanghai, Guangzhou, Shenzhen, Chengdu, Xi'an, etc.
- Individual pollutant readings: PM2.5, PM10, NO2, CO, SO2, O3
- Station-level and city-level data
- 3-8 day air quality forecasts

**API Endpoints:**
```python
# City-level AQI
http://api.waqi.info/feed/shanghai/?token=YOUR_TOKEN

# Geo-location query
http://api.waqi.info/feed/geo:31.2047;121.4489/?token=YOUR_TOKEN

# Stations within bounds
http://api.waqi.info/map/bounds/?token=YOUR_TOKEN&latlng=39.5,115.5,40.5,117.5

# Search stations
http://api.waqi.info/search/?token=YOUR_TOKEN&keyword=beijing
```

**Python Example:**
```python
import requests

TOKEN = "your_free_token"

# Get air quality for Shanghai
url = f"http://api.waqi.info/feed/shanghai/?token={TOKEN}"
response = requests.get(url)
data = response.json()

if data['status'] == 'ok':
    print(f"City: {data['data']['city']['name']}")
    print(f"AQI: {data['data']['aqi']}")
    print(f"PM2.5: {data['data']['iaqi'].get('pm25', {}).get('v', 'N/A')}")
    print(f"PM10: {data['data']['iaqi'].get('pm10', {}).get('v', 'N/A')}")
```

**Free Token Registration:**
1. Visit https://aqicn.org/data-platform/token/
2. Enter your email
3. Token is sent immediately

**Best For:** Air quality features for demand forecasting (outdoor activities, retail foot traffic, health product demand)

---

## 5. Traffic & Transportation Data

### 5.1 Gaode (AMap) API

| Attribute | Details |
|-----------|---------|
| **URL** | https://developer.amap.com |
| **Free Tier** | 300,000 requests/day for most services |
| **API Key Required** | Yes - requires Chinese phone number for registration |
| **Best For** | Urban navigation, real-time traffic, route planning |

**Available Services:**
- Real-time traffic conditions
- Traffic prediction (AI-powered 30-min forecasts)
- Route planning (driving, walking, public transit)
- Geocoding / reverse geocoding
- POI search (10M+ POIs)

**Limitations:**
- Registration requires Chinese phone number
- Advanced features require local server hosting in China
- API documentation in Chinese
- GCJ-02 coordinate system (offset from WGS-84)

**Best For:** Real-time traffic conditions, commute pattern analysis (requires Chinese entity registration)

---

### 5.2 Baidu Maps API

| Attribute | Details |
|-----------|---------|
| **URL** | https://lbsyun.baidu.com |
| **Free Tier** | Generous free tier for registered developers |
| **API Key Required** | Yes |
| **Coverage** | Best coverage in China |

**Key Features:**
- Real-time traffic data
- Route planning (driving, transit, cycling)
- Geocoding (address to coordinates)
- Place search
- Street view
- Uses GCJ-02 coordinate system (BD-09 variant)

**Limitations:**
- Individual developer registration limited to Chinese citizens
- Company accounts can be opened by overseas companies
- Requires ICP filing for commercial apps in China
- UI in Chinese only

**Best For:** Traffic congestion features, when you have Chinese business entity

---

### 5.3 PTV Flows API

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.ptvgroup.com/en-us/products/ptv-flows-api |
| **Free Tier** | Free tier available for developers |
| **Coverage** | Global including China |
| **Data Types** | Real-time, historical, and predictive traffic |

**Features:**
- Real-time traffic conditions
- Historical traffic data
- Short-term traffic forecasts (up to 60 min ahead)
- Speed profiles, congestion levels
- Corridor travel times
- Automatic map-matching

**Limitations:**
- China coverage may not be as detailed as Gaode/Baidu
- Primarily focused on major roads/highways

**Best For:** Global traffic data needs, when China-specific APIs are not accessible

---

## 6. City Classification Data

### 6.1 Yicai Global City Tier Rankings

**The most influential unofficial city classification system in China.**

| Attribute | Details |
|-----------|---------|
| **Source** | Yicai Global / CBN (第一财经) |
| **First Published** | 2016 (annual since then) |
| **Coverage** | 337 prefecture-level cities |
| **2025 Report** | https://www.yicaiglobal.com |

**2025 Tier Classification:**

| Tier | Cities | Count |
|------|--------|-------|
| **1st Tier** | Shanghai, Beijing, Shenzhen, Guangzhou | 4 |
| **New 1st Tier** | Chengdu, Hangzhou, Chongqing, Wuhan, Suzhou, Xi'an, Nanjing, Changsha, Zhengzhou, Tianjin, Hefei, Qingdao, Dongguan, Ningbo, Foshan | 15 |
| **2nd Tier** | (30 cities including Kunming, Xiamen, Jinan, Fuzhou, etc.) | 30 |
| **3rd Tier** | (70 cities) | 70 |
| **4th Tier** | (90 cities) | 90 |
| **5th Tier** | (128 cities) | 128 |

**Evaluation Dimensions (5 criteria):**
1. **Concentration of Commercial Resources Index** (brands, commerce)
2. **City as a Hub Index** (transportation, intercity connectivity)
3. **Urban Residents' Activity Index** (consumption, nightlife, lifestyle)
4. **New Economy Competitiveness Index** (tech companies, innovation)
5. **Future Potential Index** (talent attraction, innovation capacity)

**How to Use for Forecasting:**
```python
# Create a mapping of city to tier
city_tiers = {
    'Shanghai': 1, 'Beijing': 1, 'Shenzhen': 1, 'Guangzhou': 1,
    'Chengdu': 2, 'Hangzhou': 2, 'Chongqing': 2, 'Wuhan': 2,
    'Suzhou': 2, 'Xi\'an': 2, 'Nanjing': 2, 'Changsha': 2,
    'Zhengzhou': 2, 'Tianjin': 2, 'Hefei': 2, 'Qingdao': 2,
    'Dongguan': 2, 'Ningbo': 2, 'Foshan': 2,
    # ... 2nd tier cities
}

# Use as a categorical feature
df['city_tier'] = df['city_name'].map(city_tiers)
df['city_tier'] = df['city_tier'].fillna(5)  # Default to tier 5
```

**Key Insight:** Tier ranking correlates strongly with consumer spending power, retail density, and demand patterns.

---

### 6.2 GYBrand China Top 100 Cities Index

| Attribute | Details |
|-----------|---------|
| **Source** | GYBrand (城市指数研究机构) |
| **2026 Report** | Released March 2026 |
| **URL** | Mentioned in China Daily regional editions |

**Evaluation Framework (10 dimensions):**
1. Economic strength
2. Infrastructure
3. Governance efficiency
4. Medical and educational resources
5. Cultural exchange
6. Business environment
7. Development potential
8. Innovation vitality
9. Brand leadership
10. International reputation

**How to Access:**
- Annual report published in Chinese financial media
- Full list typically available via Yicai Global or GYBrand website
- PDF reports can be downloaded

**Best For:** Alternative city classification, more granular ranking within tiers

---

## 7. Summary Comparison Table

| Source | Category | Free Tier | API Key | China Coverage | Historical Data | Best For |
|--------|----------|-----------|---------|----------------|-----------------|----------|
| **Open-Meteo** | Weather | Unlimited | No | All cities via lat/lon | 1940-now | #1 choice - historical weather |
| **OpenWeatherMap** | Weather | 60/min | Yes | All major cities | 47+ yrs (paid) | Current + forecast |
| **Visual Crossing** | Weather | 1,000/day | Yes | All cities | 50+ yrs | Bulk historical download |
| **Meteostat** | Weather | Unlimited | No | Station-based | Long-term | Python/pandas workflows |
| **Jiejiari API** | Holidays | 50-100/day | Optional | China only | N/A | Chinese holidays + tiaoxiu |
| **Nager.Date** | Holidays | Unlimited | No | 100+ countries | N/A | Simple holiday API |
| **lunar_python** | Calendar | N/A (library) | N/A | N/A (offline) | N/A | Lunar calendar features |
| **chinadata.live** | Economic | Unlimited | No | National level | 1949-now | GDP, trade, population, CPI |
| **World Bank** | Economic | Unlimited | No | National level | 1960-now | Cross-country comparison |
| **WAQI** | Air Quality | 1,000/sec | Free token | 1000+ cities | Current + forecast | AQI features |
| **Gaode/AMap** | Traffic | 300K/day | Yes | Best in China | Real-time | Real-time traffic |
| **Baidu Maps** | Traffic | Generous | Yes | Best in China | Real-time | Traffic + POI |
| **Yicai Tiers** | Classification | Free | N/A | 337 cities | Annual | City tier features |

---

## 8. Implementation Recommendations

### Recommended Data Pipeline for Demand Forecasting

```python
# ============================================================
# COMPLETE FREE DATA PIPELINE FOR CHINA DEMAND FORECASTING
# ============================================================

import requests
import pandas as pd
from datetime import datetime, timedelta
from lunar_python import Solar, Lunar

# ----------------------------------------------------------
# 1. WEATHER DATA (Open-Meteo - FREE, NO KEY)
# ----------------------------------------------------------
def fetch_weather_data(lat, lon, start_date, end_date):
    """Fetch daily weather data for any Chinese city"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                 "precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max,"
                 "shortwave_radiation_sum,relative_humidity_2m_mean",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['daily'])
    df['date'] = pd.to_datetime(df['time'])
    return df

# ----------------------------------------------------------
# 2. CHINESE HOLIDAYS (Nager.Date - FREE, NO KEY)
# ----------------------------------------------------------
def fetch_china_holidays(year):
    """Fetch Chinese public holidays"""
    url = f"https://date.nager.at/api/v3/publicholidays/{year}/CN"
    response = requests.get(url)
    holidays = response.json()
    return pd.DataFrame(holidays)[['date', 'localName', 'name']]

# Combine with Jiejiari API for tiaoxiu (swap days) detail
def fetch_china_holidays_detailed(year):
    """Fetch detailed holiday info including tiaoxiu"""
    url = f"https://api.jiejiariapi.com/v1/holidays/{year}"
    response = requests.get(url)
    holidays = response.json()
    df = pd.DataFrame(holidays)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'name', 'is_off_day', 'is_holiday', 'type']]

# ----------------------------------------------------------
# 3. LUNAR CALENDAR FEATURES (lunar_python - FREE, OFFLINE)
# ----------------------------------------------------------
def add_lunar_features(df, date_col='date'):
    """Add Chinese lunar calendar features"""
    features = []
    for date in df[date_col]:
        solar = Solar.fromDate(date)
        lunar = solar.getLunar()
        features.append({
            'lunar_month': lunar.getMonth(),
            'lunar_day': lunar.getDay(),
            'is_lunar_new_year_month': lunar.getMonth() == 1,
            'is_mid_autumn_month': lunar.getMonth() == 8,
            'is_dragon_boat_month': lunar.getMonth() == 5,
            'solar_term': lunar.getJieQi(),
            'zodiac_year': lunar.getYearShengXiao(),
        })
    return pd.concat([df, pd.DataFrame(features)], axis=1)

# ----------------------------------------------------------
# 4. ECONOMIC INDICATORS (chinadata.live - FREE, NO KEY)
# ----------------------------------------------------------
def fetch_china_economic_indicator(dataset_id):
    """Fetch economic indicators from China Data Portal"""
    url = f"https://chinadata.live/api/v2/data/{dataset_id}"
    response = requests.get(url)
    data = response.json()['data']
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'])
    return df

# ----------------------------------------------------------
# 5. AIR QUALITY (WAQI - FREE TOKEN)
# ----------------------------------------------------------
def fetch_aqi_data(city, token):
    """Fetch air quality data"""
    url = f"http://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        return {
            'aqi': data['data']['aqi'],
            'pm25': data['data']['iaqi'].get('pm25', {}).get('v'),
            'pm10': data['data']['iaqi'].get('pm10', {}).get('v'),
            'no2': data['data']['iaqi'].get('no2', {}).get('v'),
            'co': data['data']['iaqi'].get('co', {}).get('v'),
        }
    return None

# ----------------------------------------------------------
# 6. CITY TIER CLASSIFICATION (static data)
# ----------------------------------------------------------
CITY_TIERS = {
    'Shanghai': 1, 'Beijing': 1, 'Shenzhen': 1, 'Guangzhou': 1,
    'Chengdu': 2, 'Hangzhou': 2, 'Chongqing': 2, 'Wuhan': 2,
    'Suzhou': 2, 'Xi\'an': 2, 'Nanjing': 2, 'Changsha': 2,
    'Zhengzhou': 2, 'Tianjin': 2, 'Hefei': 2, 'Qingdao': 2,
    'Dongguan': 2, 'Ningbo': 2, 'Foshan': 2,
    # Add 2nd, 3rd, 4th, 5th tier cities as needed
}

# ----------------------------------------------------------
# COMBINED FEATURE GENERATION
# ----------------------------------------------------------
def generate_china_forecasting_features(
    city_name, city_lat, city_lon,
    start_date, end_date,
    aqi_token=None
):
    """
    Generate complete feature set for demand forecasting
    for a Chinese city - ALL FREE DATA SOURCES
    """
    # 1. Weather features
    df = fetch_weather_data(city_lat, city_lon, start_date, end_date)
    
    # 2. Holiday features
    years = range(pd.to_datetime(start_date).year, pd.to_datetime(end_date).year + 1)
    holidays_list = []
    for year in years:
        try:
            h = fetch_china_holidays_detailed(year)
            holidays_list.append(h)
        except:
            pass
    if holidays_list:
        holidays_df = pd.concat(holidays_list)
        df = df.merge(holidays_df, on='date', how='left')
        df['is_holiday'] = df['is_holiday'].fillna(False)
        df['is_off_day'] = df['is_off_day'].fillna(False)
    
    # 3. Lunar calendar features
    df = add_lunar_features(df)
    
    # 4. City tier
    df['city_tier'] = CITY_TIERS.get(city_name, 5)
    
    # 5. Day of week / month features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # 6. Golden week flags
    df['is_spring_festival_week'] = (
        (df['month'] == 1) & (df['lunar_month'] == 1) |
        ((df['month'] == 2) & (df['lunar_month'] == 1))
    )
    df['is_national_day_week'] = (
        (df['month'] == 10) & (df['date'].dt.day >= 1) & (df['date'].dt.day <= 7)
    )
    
    return df


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Shanghai coordinates
    features = generate_china_forecasting_features(
        city_name="Shanghai",
        city_lat=31.2304,
        city_lon=121.4737,
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    print(features.head())
    print(f"Total features: {len(features.columns)}")
    print(f"Feature columns: {list(features.columns)}")
```

### Cost Summary (All Free Tier)

| Data Source | Monthly Cost | Key Requirement |
|-------------|-------------|-----------------|
| Open-Meteo Weather | $0 | None |
| Nager.Date Holidays | $0 | None |
| lunar_python Library | $0 | pip install |
| chinadata.live Economic | $0 | None |
| World Bank Data | $0 | None |
| WAQI Air Quality | $0 | Free email token |
| Jiejiari Holiday API | $0 (50-100 req/day) | Optional API key |
| **TOTAL** | **$0/month** | |

### Key Feature Categories for Demand Forecasting

| Category | Features | Data Source |
|----------|----------|-------------|
| **Temperature** | Max, min, mean temp | Open-Meteo |
| **Precipitation** | Rain, snow, total | Open-Meteo |
| **Seasonal** | Month, quarter, solar term | lunar_python |
| **Holidays** | Is holiday, is off day, holiday name | Jiejiari API |
| **Festival** | Spring Festival, National Day, etc. | lunar_python + Jiejiari |
| **Economic** | GDP growth, CPI (national level) | chinadata.live |
| **Air Quality** | AQI, PM2.5 | WAQI |
| **City Level** | Tier classification | Static data |
| **Weekend** | Is weekend, day of week | Date features |
| **Lunar** | Lunar month, lunar day | lunar_python |
