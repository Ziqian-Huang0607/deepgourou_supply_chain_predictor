"""features.py - 40+ hand-engineered features (calendar, lags, rolling, encodings)."""
from typing import Dict, List
import numpy as np
import pandas as pd
from utils import get_logger

logger = get_logger("features")


class FeatureEngineer:
    """sklearn-compatible feature transformer. fit learns stats; transform generates features."""

    def __init__(self):
        self.feature_cols: List[str] = []
        self._global_mean = 0.0
        self._cust_means: Dict = {}
        self._cat_means: Dict = {}
        self._cc_means: Dict = {}
        self._prov_means: Dict = {}
        self._dow_means: Dict = {}

    def fit(self, df: pd.DataFrame):
        q = df["quantity"]
        self._global_mean = q.mean()
        self._cust_means = df.groupby("customer_code")["quantity"].mean().to_dict()
        self._cat_means = df.groupby("product_category")["quantity"].mean().to_dict()
        self._cc_means = df.groupby(["customer_code", "product_category"])["quantity"].mean().to_dict()
        if "province" in df.columns:
            self._prov_means = df.groupby("province")["quantity"].mean().to_dict()
        self._dow_means = df.groupby(df["date"].dt.dayofweek)["quantity"].mean().to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["customer_code", "product_category", "date"])
        gc = ["customer_code", "product_category"]

        # ---- 1. Temporal ----
        df["dow"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["is_mon"] = (df["dow"] == 0).astype(int)
        df["is_fri"] = (df["dow"] == 4).astype(int)
        df["is_wed"] = (df["dow"] == 2).astype(int)

        # Cyclical encoding
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
        df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

        # ---- 2. Calendar features (from external_data) ----
        for col in ["is_holiday", "days_to_holiday", "days_from_holiday",
                    "is_lny_period", "is_qingming_period", "is_labor_period",
                    "is_solar_term", "days_to_solar", "days_from_solar"]:
            if col not in df.columns:
                df[col] = 0

        # ---- 3. Lag features (shifted = NO LEAKAGE) ----
        df["lag_1"] = df.groupby(gc)["quantity"].shift(1).fillna(0)
        df["lag_2"] = df.groupby(gc)["quantity"].shift(2).fillna(0)
        df["lag_7"] = df.groupby(gc)["quantity"].shift(7).fillna(0)
        df["lag_14"] = df.groupby(gc)["quantity"].shift(14).fillna(0)
        df["lag_21"] = df.groupby(gc)["quantity"].shift(21).fillna(0)

        # ---- 4. Rolling statistics (PAST only, shift 1) ----
        gq = df.groupby(gc)["quantity"]
        df["rm7"] = gq.shift(1).rolling(7, min_periods=1).mean().fillna(0)
        df["rm14"] = gq.shift(1).rolling(14, min_periods=1).mean().fillna(0)
        df["rm21"] = gq.shift(1).rolling(21, min_periods=1).mean().fillna(0)
        df["rs7"] = gq.shift(1).rolling(7, min_periods=1).std().fillna(0)
        df["rmax7"] = gq.shift(1).rolling(7, min_periods=1).max().fillna(0)
        df["rmin7"] = gq.shift(1).rolling(7, min_periods=1).min().fillna(0)

        # Rolling counts of non-zero days
        df["rnz7"] = gq.shift(1).rolling(7, min_periods=1).apply(lambda x: (x > 0).sum()).fillna(0)

        # ---- 5. Expanding mean encodings ----
        df["enc_cust"] = df["customer_code"].map(self._cust_means).fillna(self._global_mean)
        df["enc_cat"] = df["product_category"].map(self._cat_means).fillna(self._global_mean)
        df["enc_cc"] = df.apply(
            lambda r: self._cc_means.get((r["customer_code"], r["product_category"]),
                                         self._cust_means.get(r["customer_code"], self._global_mean)), axis=1)
        if self._prov_means:
            df["enc_prov"] = df.get("province", "").map(self._prov_means).fillna(self._global_mean)
        else:
            df["enc_prov"] = self._global_mean
        df["enc_dow"] = df["dow"].map(self._dow_means).fillna(self._global_mean)

        # ---- 6. Hierarchy aggregations ----
        cd = df.groupby(["date", "customer_code"])["quantity"].sum().reset_index()
        cd.columns = ["date", "customer_code", "cust_total"]
        df = df.merge(cd, on=["date", "customer_code"], how="left")

        cad = df.groupby(["date", "product_category"])["quantity"].sum().reset_index()
        cad.columns = ["date", "product_category", "cat_total"]
        df = df.merge(cad, on=["date", "product_category"], how="left")

        # ---- 7. Ratio features ----
        df["share_in_cust"] = (df["quantity"] / df["cust_total"].replace(0, np.nan)).fillna(0)

        # ---- 8. Trend ----
        df["day_num"] = (df["date"] - df["date"].min()).dt.days
        df["month_day"] = df["month"] * 100 + df["day"]

        # ---- Feature list ----
        self.feature_cols = [
            "dow", "month", "day", "week", "quarter",
            "is_weekend", "is_month_start", "is_month_end",
            "is_mon", "is_fri", "is_wed",
            "dow_sin", "dow_cos", "month_sin", "month_cos", "day_sin", "day_cos",
            "is_holiday", "days_to_holiday", "days_from_holiday",
            "is_lny_period", "is_qingming_period", "is_labor_period",
            "is_solar_term", "days_to_solar", "days_from_solar",
            "lag_1", "lag_2", "lag_7", "lag_14", "lag_21",
            "rm7", "rm14", "rm21", "rs7", "rmax7", "rmin7", "rnz7",
            "enc_cust", "enc_cat", "enc_cc", "enc_prov", "enc_dow",
            "cust_total", "cat_total", "share_in_cust",
            "day_num", "month_day",
        ]
        # Only keep columns that exist and are numeric
        self.feature_cols = [c for c in self.feature_cols
                             if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_feature_cols(self) -> List[str]:
        return self.feature_cols.copy()
