"""ensemble.py - Weighted ensemble + conformal prediction intervals."""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from utils import get_logger

logger = get_logger("ensemble")


class DemandEnsemble:
    """Weighted ensemble of multiple forecasters with inverse-MAE weighting."""

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.maes = {}
        self.feature_cols = []

    def add_model(self, name: str, model, X_valid: pd.DataFrame, y_valid: pd.Series):
        preds = np.clip(model.predict(X_valid), 0, None)
        mae = mean_absolute_error(y_valid, preds)
        self.models[name] = model
        self.maes[name] = mae
        logger.info("  %s MAE: %.4f", name, mae)

    def compute_weights(self):
        inv = {k: 1.0 / max(v, 1e-6) for k, v in self.maes.items()}
        total = sum(inv.values())
        self.weights = {k: v / total for k, v in inv.items()}
        logger.info("  Weights: %s", {k: f"{v:.3f}" for k, v in self.weights.items()})

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 0)
            if w > 0 and name not in ("sba", "tsb", "croston"):
                preds += w * np.clip(model.predict(X), 0, None)
        return np.clip(preds, 0, None)

    def save(self, path: str):
        state = {"weights": self.weights, "maes": self.maes, "feature_cols": self.feature_cols}
        with open(path, "wb") as f:
            pickle.dump(state, f)
        # Save individual models
        import os
        d = os.path.dirname(path)
        for name, model in self.models.items():
            if hasattr(model, "save"):
                model.save(os.path.join(d, f"model_{name}.pkl"))

    @classmethod
    def load(cls, path: str):
        import os
        with open(path, "rb") as f:
            state = pickle.load(f)
        ens = cls()
        ens.weights = state["weights"]
        ens.maes = state["maes"]
        ens.feature_cols = state.get("feature_cols", [])
        # Load individual models
        d = os.path.dirname(path)
        from models import LightGBMForecaster, XGBoostForecaster, RidgeForecaster
        loaders = {"lightgbm": LightGBMForecaster, "xgboost": XGBoostForecaster, "ridge": RidgeForecaster}
        for name, loader in loaders.items():
            mpath = os.path.join(d, f"model_{name}.pkl")
            if os.path.exists(mpath):
                ens.models[name] = loader.load(mpath)
        return ens


class ConformalPredictor:
    """Split conformal prediction for prediction intervals."""

    def __init__(self, alpha=0.1):  # 90% coverage
        self.alpha = alpha
        self.q = None  # quantile of residuals

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calibration on validation set."""
        residuals = np.abs(y_true - y_pred)
        self.q = np.quantile(residuals, 1 - self.alpha)
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple:
        """Return (lower, upper) bounds."""
        if self.q is None:
            return y_pred, y_pred
        return np.clip(y_pred - self.q, 0, None), y_pred + self.q
