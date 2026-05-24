#!/usr/bin/env python3
"""
main.py - HKU AI Competition Problem 2: Award-Winning Demand Forecasting

Multi-model ensemble (LightGBM + XGBoost + SBA + Ridge) with:
- External data integration (holidays, solar terms, optional weather)
- Generic year support (works for ANY year)
- Robust schema detection (works for ANY data format)
- Conformal prediction intervals
- DOW-based daily distribution

Usage:
    python main.py --train-data "jan.xlsx" "feb_mar.xlsx"
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, MODEL_DIR
from data_loader import load_excel_auto, categorize_products, aggregate_daily
from external_data import merge_external_features
from features import FeatureEngineer
from intermittent import classify_demand
from train import train_pipeline
from predict import generate_predictions, save_submission
from utils import get_logger

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="HKU AI Competition Problem 2")
    parser.add_argument("--train-data", nargs="+", required=True, help="Order Excel files")
    parser.add_argument("--predict-start", default="2026-04-01", help="Prediction start")
    parser.add_argument("--predict-end", default="2026-04-30", help="Prediction end")
    parser.add_argument("--use-weather", action="store_true", help="Fetch weather (slow)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    start = time.time()

    logger.info("=" * 60)
    logger.info("HKU AI Competition - Problem 2: Demand Forecasting")
    logger.info("Models: LightGBM + XGBoost + SBA + Ridge")
    logger.info("Features: Calendar + Holidays + Solar Terms + Lags + Encodings")
    logger.info("Generic: Any year, any data format, robust schema detection")
    logger.info("=" * 60)

    # Step 1: Load data with auto schema detection
    logger.info("\n[Step 1] Loading data...")
    all_raw = []
    for fp in args.train_data:
        df = load_excel_auto(fp)
        all_raw.append(df)
    raw = pd.concat(all_raw, ignore_index=True)
    raw = categorize_products(raw)

    # Detect year from data
    year = int(raw["_year"].mode()[0]) if "_year" in raw.columns else 2026
    logger.info("Detected year: %d", year)

    # Step 2: Aggregate to daily
    logger.info("\n[Step 2] Daily aggregation...")
    daily = aggregate_daily(raw, end_date=args.predict_start)

    # Step 3: Classify demand patterns
    logger.info("\n[Step 3] Demand classification...")
    patterns = classify_demand(daily)
    logger.info("Patterns:")
    for _, r in patterns.iterrows():
        if r["n_nonzero"] > 0:
            logger.info("  %s | %-20s | ADI=%.2f CV2=%.2f | %s",
                        r["customer_code"], r["product_category"], r["adi"], r["cv2"], r["pattern"])

    # Step 4: External data (holidays + solar terms + optional weather)
    logger.info("\n[Step 4] External features...")
    daily = merge_external_features(daily, use_weather=args.use_weather)

    # Step 5: Feature engineering
    logger.info("\n[Step 5] Feature engineering...")
    fe = FeatureEngineer()
    featured = fe.fit_transform(daily)
    logger.info("Features: %d", len(fe.get_feature_cols()))

    # Step 6: Train
    logger.info("\n[Step 6] Training...")
    ensemble, feat_cols, conformal = train_pipeline(featured)

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensemble.save(os.path.join(MODEL_DIR, "ensemble.pkl"))
    logger.info("Models saved.")

    # Step 7: Predict
    logger.info("\n[Step 7] Predicting...")
    predictions = generate_predictions(ensemble, fe, daily, args.predict_start, args.predict_end)

    # Step 8: Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub = save_submission(predictions, os.path.join(OUTPUT_DIR, "april_2026_predictions.csv"))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("Total: %d predictions", len(sub))
    nz = (sub["predicted_quantity"] > 0).sum()
    logger.info("Non-zero: %d (%.1f%%)", nz, 100 * nz / len(sub))
    logger.info("Grand total: %.2f", sub["predicted_quantity"].sum())
    logger.info("\nBy customer:")
    for c, g in sub.groupby("customer_code"):
        logger.info("  %s: %.2f", c, g["predicted_quantity"].sum())
    logger.info("\nBy category (top 5):")
    for cat, val in sub.groupby("product_category")["predicted_quantity"].sum().sort_values(ascending=False).head(5).items():
        logger.info("  %-20s: %.2f", cat, val)
    logger.info("\nDone in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
