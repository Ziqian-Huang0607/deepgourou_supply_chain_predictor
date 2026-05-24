"""predict.py - Generate predictions with DOW distribution + guardrails."""
import numpy as np
import pandas as pd
from config import MAX_SCALE, MIN_SCALE
from features import FeatureEngineer
from utils import get_logger

logger = get_logger("predict")


def compute_dow_weights(history: pd.DataFrame) -> dict:
    """Learn DOW proportions per customer x category."""
    history["dow"] = history["date"].dt.dayofweek
    dow = history.groupby(["customer_code", "product_category", "dow"])["quantity"].sum().reset_index()
    weights = {}
    for (c, cat), g in dow.groupby(["customer_code", "product_category"]):
        props = np.zeros(7)
        for _, r in g.iterrows():
            props[int(r["dow"])] = r["quantity"]
        total = props.sum()
        weights[(c, cat)] = props / total if total > 0 else np.ones(7) / 7
    return weights


def generate_predictions(ensemble, fe: FeatureEngineer, train_df: pd.DataFrame,
                         predict_start: str, predict_end: str) -> pd.DataFrame:
    """Generate predictions with DOW distribution + monthly guardrails."""
    logger.info("=" * 50)
    logger.info("PREDICTING %s to %s", predict_start, predict_end)
    logger.info("=" * 50)

    dates = pd.date_range(predict_start, predict_end, freq="D")
    customers = train_df["customer_code"].unique()
    categories = train_df["product_category"].unique()

    # Historical stats for guardrails
    hist = {}
    for (c, cat), g in train_df.groupby(["customer_code", "product_category"]):
        q = g["quantity"].values
        hist[(c, cat)] = {"max": q.max(), "mean": q.mean(), "sum": q.sum(), "std": q.std()}

    # DOW weights
    dow_w = compute_dow_weights(train_df)

    # Build prediction grid
    grid = pd.MultiIndex.from_product([dates, customers, categories],
                                       names=["date", "customer_code", "product_category"]).to_frame(index=False)
    for col in ["province", "warehouse"]:
        if col in train_df.columns:
            grid[col] = grid["customer_code"].map(train_df.groupby("customer_code")[col].first().to_dict())
    grid["quantity"] = 0
    grid["year"] = grid["date"].dt.year

    # Combine with history
    base_cols = ["date", "customer_code", "product_category", "quantity", "province", "warehouse", "year"]
    base_cols = [c for c in base_cols if c in train_df.columns]
    train_base = train_df[base_cols].copy()
    combined = pd.concat([train_base, grid], ignore_index=True)
    combined = combined.sort_values(["customer_code", "product_category", "date"]).reset_index(drop=True)

    # Engineer features
    featured = fe.transform(combined)
    pred_mask = featured["date"] >= predict_start
    pred_df = featured[pred_mask].copy()

    # Fill zero lags with historical mean
    for col in ["lag_1", "lag_2", "lag_7", "lag_14", "lag_21", "rm7", "rm14", "rm21"]:
        pred_df[col] = pred_df.apply(
            lambda r: hist.get((r["customer_code"], r["product_category"]), {}).get("mean", 0)
            if r[col] == 0 else r[col], axis=1)

    # Get LGBM + XGB + Ridge predictions (use exact feature cols from training)
    feat_cols = fe.get_feature_cols()
    X_pred = pred_df[feat_cols].fillna(0).values  # Convert to numpy to avoid feature name issues
    lgb_p = np.clip(ensemble.models["lightgbm"].predict(X_pred), 0, None)
    xgb_p = np.clip(ensemble.models["xgboost"].predict(X_pred), 0, None) if "xgboost" in ensemble.models else np.zeros(len(X_pred))
    ridge_p = np.clip(ensemble.models["ridge"].predict(X_pred), 0, None)

    # SBA daily predictions
    sba = ensemble.models.get("sba")
    sba_p = np.zeros(len(pred_df))
    if sba:
        for i, (_, r) in enumerate(pred_df.iterrows()):
            sba_p[i] = sba.predict_monthly(r["customer_code"], r["product_category"], 1)

    # Weighted base predictions
    w = ensemble.weights
    base = (w.get("lightgbm", 0.4) * lgb_p +
            w.get("xgboost", 0.2) * xgb_p +
            w.get("ridge", 0.1) * ridge_p +
            w.get("sba", 0.3) * sba_p)

    # ADIDA: Compute monthly total from historical data + trend, then DOW distribute
    final = np.zeros(len(pred_df))
    pos = 0
    for (cust, cat), group in pred_df.groupby(["customer_code", "product_category"], sort=False):
        n = len(group)
        idxs = list(range(pos, pos + n))
        stats = hist.get((cust, cat), {})

        if stats.get("sum", 0) <= 0:
            pos += n
            continue

        # Robust monthly: historical daily mean * 30, with conservative bounds
        hist_monthly = stats["mean"] * 30
        # Pure historical with small growth (no LGBM blending - too unreliable)
        monthly = hist_monthly * 1.05  # 5% growth trend
        # Tight guardrails
        monthly = min(monthly, hist_monthly * 1.5)
        monthly = max(monthly, hist_monthly * 0.6)

        # Distribute monthly total across ~30 days with DOW variation
        # daily_avg = monthly / 30, scaled by DOW preference (dp[day] * 7)
        # This preserves monthly total: sum over 30 days = monthly
        dp = dow_w.get((cust, cat), np.ones(7) / 7).copy()
        dp = dp + 0.1  # Laplace smoothing
        dp = dp / dp.sum()
        daily_avg = monthly / 30.0
        for gi, idx in enumerate(idxs):
            day = pred_df.iloc[idx]["date"].dayofweek
            final[idx] = daily_avg * (dp[day] * 7)  # Scale by DOW preference
        pos += n

    pred_df["predicted_quantity"] = np.round(np.clip(final, 0, None), 2)

    n_nz = (pred_df["predicted_quantity"] > 0.01).sum()
    logger.info("Predictions: %d, Non-zero: %d (%.1f%%), Total: %.2f",
                len(pred_df), n_nz, 100 * n_nz / len(pred_df), pred_df["predicted_quantity"].sum())
    return pred_df


def save_submission(pred_df: pd.DataFrame, path: str):
    sub = pred_df[["date", "customer_code", "product_category", "predicted_quantity"]].copy()
    sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
    # Deduplicate: group by date×customer×category and sum
    sub = sub.groupby(["date", "customer_code", "product_category"], as_index=False)["predicted_quantity"].sum()
    sub.to_csv(path, index=False)
    logger.info("Saved: %s (%d rows)", path, len(sub))
    return sub
