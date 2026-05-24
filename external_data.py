"""external_data.py - Weather, holidays, calendar integration. Generic any year."""
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import CACHE_DIR, CITY_COORDS
from utils import get_logger

logger = get_logger("external_data")


class HolidayEngine:
    """Chinese holidays for any year. Uses holidays library + manual overrides."""

    # Manual Chinese holidays (month, day) -> name for any year
    FIXED_HOLIDAYS = {
        (1, 1): "new_year",
        (5, 1): "labor_day",
        (10, 1): "national_day",
        (10, 2): "national_day2",
        (10, 3): "national_day3",
    }

    def __init__(self, year: int):
        self.year = year
        self._holidays = self._build_holidays()

    def _build_holidays(self) -> Dict[str, str]:
        """Build holiday map for the year."""
        result = {}

        # Try python-holidays library first (best for dynamic years)
        try:
            import holidays
            cn_holidays = holidays.China(years=self.year)
            for date_obj, name in cn_holidays.items():
                result[date_obj.strftime("%Y-%m-%d")] = name
            logger.info("  Loaded %d holidays from python-holidays for %d", len(result), self.year)
        except ImportError:
            logger.info("  python-holidays not installed, using manual calendar")
        except Exception as e:
            logger.info("  python-holidays error: %s, using manual", e)

        # Add fixed holidays
        for (month, day), name in self.FIXED_HOLIDAYS.items():
            key = f"{self.year}-{month:02d}-{day:02d}"
            if key not in result:
                result[key] = name

        # Add known Lunar New Year dates for recent years
        lny_dates = {
            2024: "02-10", 2025: "01-29", 2026: "02-17", 2027: "02-06",
            2028: "01-26", 2029: "02-13", 2030: "02-03",
        }
        if self.year in lny_dates:
            result[f"{self.year}-{lny_dates[self.year]}"] = "lunar_new_year"
            # LNY period
            base = datetime.strptime(f"{self.year}-{lny_dates[self.year]}", "%Y-%m-%d")
            for i in range(1, 8):
                d = (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                result[d] = f"lny_day{i+1}"

        # Add Qingming (approx Apr 4-5)
        qingming_dates = {
            2024: "04-04", 2025: "04-04", 2026: "04-05", 2027: "04-05",
            2028: "04-04", 2029: "04-04", 2030: "04-05",
        }
        if self.year in qingming_dates:
            result[f"{self.year}-{qingming_dates[self.year]}"] = "qingming"

        # Add Dragon Boat (approx mid-June)
        db_dates = {
            2024: "06-10", 2025: "05-31", 2026: "06-19", 2027: "06-09",
            2028: "05-28", 2029: "06-16", 2030: "06-12",
        }
        if self.year in db_dates:
            result[f"{self.year}-{db_dates[self.year]}"] = "dragon_boat"

        # Add Mid-Autumn (approx mid-Sep to early Oct)
        ma_dates = {
            2024: "09-17", 2025: "10-06", 2026: "09-25", 2027: "09-15",
            2028: "10-03", 2029: "09-22", 2030: "09-12",
        }
        if self.year in ma_dates:
            result[f"{self.year}-{ma_dates[self.year]}"] = "mid_autumn"

        logger.info("  Total holidays for %d: %d", self.year, len(result))
        return result

    def create_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create holiday-related features."""
        df = pd.DataFrame({"date": dates})
        date_strs = dates.strftime("%Y-%m-%d")

        df["is_holiday"] = date_strs.isin(self._holidays).astype(int)
        df["holiday_name"] = date_strs.map(lambda d: self._holidays.get(d, "none"))

        # Days to next holiday
        holiday_dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in self._holidays.keys()])
        df["days_to_holiday"] = dates.map(
            lambda d: min((h - d).days for h in holiday_dates if h >= d) if holiday_dates else 365
        )
        df["days_from_holiday"] = dates.map(
            lambda d: min((d - h).days for h in holiday_dates if h <= d) if holiday_dates else 365
        )

        # Specific holiday flags
        df["is_lny_period"] = ((dates.month == 1) & (dates.day >= 25)).astype(int) + \
                               ((dates.month == 2) & (dates.day <= 23)).astype(int)
        df["is_qingming_period"] = ((dates.month == 4) & (dates.day >= 3) & (dates.day <= 6)).astype(int)
        df["is_labor_period"] = ((dates.month == 4) & (dates.day >= 28)).astype(int) + \
                                 ((dates.month == 5) & (dates.day <= 3)).astype(int)

        return df


class WeatherAPI:
    """Open-Meteo weather API - FREE, no key, global coverage."""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self):
        self._cache_dir = os.path.join(CACHE_DIR, "weather")
        os.makedirs(self._cache_dir, exist_ok=True)

    def _cache_path(self, city: str, start: str, end: str) -> str:
        return os.path.join(self._cache_dir, f"{city}_{start}_{end}.pkl")

    def fetch(self, city: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch daily weather for a city. Returns DataFrame with temp, precip, etc."""
        cache = self._cache_path(city, start, end)
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return pickle.load(f)

        coords = CITY_COORDS.get(city)
        if not coords:
            logger.info("  No coords for city: %s", city)
            return None

        try:
            import requests
            params = {
                "latitude": coords[0], "longitude": coords[1],
                "start_date": start, "end_date": end,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "Asia/Shanghai",
            }
            r = requests.get(self.BASE_URL, params=params, timeout=30)
            data = r.json()

            if "daily" not in data:
                return None

            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                f"temp_max_{city}": data["daily"]["temperature_2m_max"],
                f"temp_min_{city}": data["daily"]["temperature_2m_min"],
                f"precip_{city}": data["daily"]["precipitation_sum"],
            })

            with open(cache, "wb") as f:
                pickle.dump(df, f)
            return df

        except Exception as e:
            logger.info("  Weather fetch error for %s: %s", city, e)
            return None

    def fetch_for_cities(self, cities: list, start: str, end: str) -> pd.DataFrame:
        """Fetch weather for multiple cities and merge."""
        dfs = []
        for city in cities:
            df = self.fetch(city, start, end)
            if df is not None:
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on="date", how="outer")
        return result


class SolarTerms:
    """24 Solar Terms (节气) features. Approximate dates for any year."""

    # Approximate month/day for each solar term
    TERMS = {
        "lichun": (2, 4), "yushui": (2, 19), "jingzhe": (3, 6),
        "chunfen": (3, 21), "qingming": (4, 5), "guyu": (4, 20),
        "lixia": (5, 6), "xiaoman": (5, 21), "mangzhong": (6, 6),
        "xiazhi": (6, 21), "xiaoshu": (7, 7), "dashu": (7, 23),
        "liqiu": (8, 8), "chushu": (8, 23), "bailu": (9, 8),
        "qiufen": (9, 23), "hanlu": (10, 8), "shuangjiang": (10, 23),
        "lidong": (11, 7), "xiaoxue": (11, 22), "daxue": (12, 7),
        "dongzhi": (12, 22), "xiaohan": (1, 6), "dahan": (1, 20),
    }

    def create_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        df = pd.DataFrame({"date": dates})
        year = dates[0].year if len(dates) > 0 else 2026

        # Map solar term dates for this year
        term_dates = set()
        for name, (month, day) in self.TERMS.items():
            term_dates.add(f"{year}-{month:02d}-{day:02d}")

        df["is_solar_term"] = dates.strftime("%Y-%m-%d").isin(term_dates).astype(int)

        # Days to next/previous solar term
        all_term_dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in term_dates])
        def days_to(d):
            future = [(t - d).days for t in all_term_dates if t >= d]
            return min(future) if future else 365
        def days_from(d):
            past = [(d - t).days for t in all_term_dates if t <= d]
            return min(past) if past else 365
        df["days_to_solar"] = dates.map(days_to)
        df["days_from_solar"] = dates.map(days_from)

        return df


def merge_external_features(daily_df: pd.DataFrame, use_weather: bool = False) -> pd.DataFrame:
    """Merge all external features into daily dataframe."""
    df = daily_df.copy()
    dates = pd.DatetimeIndex(df["date"].unique())

    if len(dates) == 0:
        return df

    year = int(dates[0].year)

    # Holiday features
    logger.info("Adding holiday features for year %d...", year)
    he = HolidayEngine(year)
    hfeat = he.create_features(dates)
    df = df.merge(hfeat, on="date", how="left")

    # Solar term features
    logger.info("Adding solar term features...")
    st = SolarTerms()
    sfeat = st.create_features(dates)
    df = df.merge(sfeat, on="date", how="left")

    # Weather features (optional - requires network)
    if use_weather:
        logger.info("Fetching weather data...")
        wapi = WeatherAPI()
        # Get top cities from data
        cities_in_data = df["province"].dropna().unique() if "province" in df.columns else []
        cities_to_fetch = [c for c in cities_in_data if c in CITY_COORDS][:5]  # Max 5 cities
        if cities_to_fetch:
            start = dates.min().strftime("%Y-%m-%d")
            end = dates.max().strftime("%Y-%m-%d")
            wfeat = wapi.fetch_for_cities(cities_to_fetch, start, end)
            if len(wfeat) > 0:
                df = df.merge(wfeat, on="date", how="left")
                logger.info("  Merged weather for %d cities", len(cities_to_fetch))

    logger.info("External features merged. Shape: %s", df.shape)
    return df
