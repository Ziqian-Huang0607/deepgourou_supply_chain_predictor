"""train.py - Training pipeline with chronological split."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from features import FeatureEngineer
from models import LightGBMForecaster, XGBoostForecaster, RidgeForecaster
from intermittent import SBAModel
from ensemble import DemandEnsemble, ConformalPredictor
from utils import get_logger

logger = get_logger("train")


def chron_split(df: pd.DataFrame, n_valid: int = 7):
    """Chronological split: last N days for validation."""
    cutoff = df["date"].max() - pd.Timedelta(days=n_valid - 1)
    train = df[df["date"] < cutoff].copy()
    valid = df[df["date"] >= cutoff].copy()
    logger.info("Train: %d (%s to %s)", len(train), train["date"].min().date(), train["date"].max().date())
    logger.info("Valid: %d (%s to %s)", len(valid), valid["date"].min().date(), valid["date"].max().date())
    return train, valid


def get_feat_cols(df: pd.DataFrame) -> list:
    """Get numeric feature columns excluding target and IDs."""
    exclude = ["quantity", "date", "customer_code", "product_category",
               "province", "warehouse", "year", "holiday_name"]
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train_pipeline(featured: pd.DataFrame) -> tuple:
    """Train all models and ensemble. Returns (ensemble, feature_engineer, feat_cols, conformal)."""
    logger.info("=" * 50)
    logger.info("TRAINING")
    logger.info("=" * 50)

    train, valid = chron_split(featured)
    feat_cols = get_feat_cols(train)
    logger.info("Features: %d", len(feat_cols))

    X_tr, y_tr = train[feat_cols].fillna(0), train["quantity"].fillna(0)
    X_va, y_va = valid[feat_cols].fillna(0), valid["quantity"].fillna(0)

    # 1. LightGBM
    logger.info("\n[1/4] LightGBM")
    lgbm = LightGBMForecaster()
    lgbm.fit(X_tr, y_tr, X_va, y_va)
    try:
        top_feats = lgbm.feature_importance()[:5]
        logger.info("  Top: %s", [(n, f"{v:.0f}") for n, v in top_feats])
    except Exception:
        pass

    # 2. XGBoost (optional - skip if not installed)
    try:
        logger.info("\n[2/4] XGBoost")
        xgb = XGBoostForecaster()
        xgb.fit(X_tr, y_tr, X_va, y_va)
    except Exception as e:
        logger.info("  XGBoost skipped: %s", e)
        xgb = None

    # 3. Ridge
    logger.info("\n[3/4] Ridge")
    ridge = RidgeForecaster()
    ridge.fit(X_tr, y_tr)

    # 4. SBA
    logger.info("\n[4/4] SBA (intermittent)")
    sba = SBAModel()
    sba.fit(train)

    # Ensemble
    logger.info("\n[Ensemble]")
    ensemble = DemandEnsemble()
    ensemble.add_model("lightgbm", lgbm, X_va, y_va)
    if xgb is not None:
        ensemble.add_model("xgboost", xgb, X_va, y_va)
    ensemble.add_model("ridge", ridge, X_va, y_va)
    ensemble.models["sba"] = sba
    # SBA predictions for weighting
    sba_preds = np.array([sba.predict_monthly(r["customer_code"], r["product_category"], 1)
                          for _, r in valid.iterrows()])
    sba_preds = np.clip(sba_preds, 0, None)
    ensemble.maes["sba"] = mean_absolute_error(y_va, sba_preds)
    logger.info("  sba MAE: %.4f", ensemble.maes["sba"])
    ensemble.compute_weights()

    # Conformal prediction intervals
    ens_preds = ensemble.predict(X_va)
    conformal = ConformalPredictor(alpha=0.1)
    conformal.fit(y_va.values, ens_preds)
    logger.info("  Conformal q(90%%): %.4f", conformal.q)

    # Evaluate
    ens_mae = mean_absolute_error(y_va, ens_preds)
    logger.info("\nEnsemble MAE: %.4f", ens_mae)

    return ensemble, feat_cols, conformal
