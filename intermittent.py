"""intermittent.py - SBA, TSB, Croston for intermittent demand + classification."""
import numpy as np
import pandas as pd
from utils import get_logger

logger = get_logger("intermittent")


class CrostonModel:
    """Croston's method: separate forecasting of demand size and demand interval."""

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._p = {}  # smoothed interval
        self._z = {}  # smoothed demand size

    def fit(self, df: pd.DataFrame):
        for (cust, cat), g in df.groupby(["customer_code", "product_category"]):
            q = g.sort_values("date")["quantity"].values
            nz = np.where(q > 0)[0]
            if len(nz) == 0:
                self._p[(cust, cat)] = 999
                self._z[(cust, cat)] = 0
                continue
            p = nz[0] + 1
            z = q[nz[0]]
            for i in range(1, len(nz)):
                gap = nz[i] - nz[i - 1]
                p = self.alpha * gap + (1 - self.alpha) * p
                z = self.alpha * q[nz[i]] + (1 - self.alpha) * z
            self._p[(cust, cat)] = p
            self._z[(cust, cat)] = z
        return self

    def predict_monthly(self, cust: str, cat: str, n_days: int = 30) -> float:
        p = self._p.get((cust, cat), 999)
        z = self._z.get((cust, cat), 0)
        return z / p * n_days if p > 0 else 0


class SBAModel(CrostonModel):
    """Syntetos-Boylan Approximation: Croston with bias correction."""

    def predict_monthly(self, cust: str, cat: str, n_days: int = 30) -> float:
        p = self._p.get((cust, cat), 999)
        z = self._z.get((cust, cat), 0)
        return (1 - self.alpha / 2) * z / p * n_days if p > 0 else 0


class TSBModel:
    """Teunter-Syntetos-Babai: probability-based approach."""

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._prob = {}
        self._demand = {}

    def fit(self, df: pd.DataFrame):
        for (cust, cat), g in df.groupby(["customer_code", "product_category"]):
            q = g.sort_values("date")["quantity"].values
            is_demand = (q > 0).astype(float)
            p = is_demand[0] if len(is_demand) > 0 else 0
            z = q[0] if q[0] > 0 else 0
            for i in range(1, len(is_demand)):
                p = self.alpha * is_demand[i] + (1 - self.alpha) * p
                if q[i] > 0:
                    z = self.alpha * q[i] + (1 - self.alpha) * z
            self._prob[(cust, cat)] = p
            self._demand[(cust, cat)] = z
        return self

    def predict_monthly(self, cust: str, cat: str, n_days: int = 30) -> float:
        p = self._prob.get((cust, cat), 0)
        z = self._demand.get((cust, cat), 0)
        return p * z * n_days if p > 0 else 0


def classify_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each customer x category by ADI and CV2."""
    rows = []
    for (cust, cat), g in df.groupby(["customer_code", "product_category"]):
        q = g["quantity"].values
        nz = q[q > 0]
        adi = len(q) / max(len(nz), 1)
        cv2 = (np.std(nz) / max(np.mean(nz), 0.01)) ** 2 if len(nz) > 1 else 0

        if adi < 1.32 and cv2 < 0.49:
            pattern = "smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            pattern = "intermittent"
        elif adi >= 1.32 and cv2 >= 0.49:
            pattern = "lumpy"
        else:
            pattern = "erratic"

        rows.append({"customer_code": cust, "product_category": cat,
                     "adi": adi, "cv2": cv2, "pattern": pattern,
                     "n_nonzero": len(nz), "mean_qty": nz.mean() if len(nz) > 0 else 0})
    return pd.DataFrame(rows)
