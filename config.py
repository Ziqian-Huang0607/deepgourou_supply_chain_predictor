"""config.py - Generic configuration (works for ANY year, ANY data)."""
import os
from typing import Dict, Tuple

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, ".cache")
for d in [OUTPUT_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# Model params
LGBM_PARAMS = {
    "objective": "tweedie", "tweedie_variance_power": 1.5,
    "metric": "mae", "boosting_type": "gbdt",
    "num_leaves": 63, "max_depth": 8, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "min_child_samples": 5,
    "verbose": -1, "seed": 42,
}
LGBM_ROUNDS, LGBM_EARLY = 2000, 50

XGB_PARAMS = {
    "objective": "reg:tweedie", "tweedie_variance_power": 1.5,
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 5,
    "eval_metric": "mae", "seed": 42, "verbosity": 0,
}
XGB_ROUNDS, XGB_EARLY = 2000, 50

RIDGE_ALPHA = 10.0
SBA_ALPHA = 0.15

# Guardrails
MAX_SCALE = 2.5
MIN_SCALE = 0.5

# City coordinates for weather lookup (lat, lon)
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "深圳": (22.54, 114.06), "武汉": (30.59, 114.31),
    "广州": (23.13, 113.26), "合肥": (31.82, 117.23),
    "杭州": (30.27, 120.15), "成都": (30.57, 104.07),
    "西安": (34.34, 108.94), "上海": (31.23, 121.47),
    "北京": (39.90, 116.41), "南京": (32.06, 118.78),
    "苏州": (31.30, 120.58), "宁波": (29.87, 121.55),
    "佛山": (23.02, 113.11), "东莞": (23.05, 113.75),
    "南宁": (22.82, 108.32), "贵阳": (26.58, 106.71),
    "长沙": (28.23, 112.98), "昆明": (25.04, 102.71),
    "郑州": (34.75, 113.65), "太原": (37.87, 112.55),
    "济南": (36.65, 116.98), "青岛": (36.07, 120.38),
    "大连": (38.92, 121.62), "沈阳": (41.80, 123.43),
    "长春": (43.82, 125.32), "哈尔滨": (45.80, 126.53),
    "石家庄": (38.04, 114.51), "福州": (26.08, 119.30),
    "厦门": (24.48, 118.09), "海口": (20.02, 110.35),
    "呼和浩特": (40.84, 111.75), "乌鲁木齐": (43.83, 87.62),
    "兰州": (36.06, 103.83), "银川": (38.49, 106.23),
    "西宁": (36.62, 101.78), "拉萨": (29.65, 91.13),
    "无锡": (31.49, 120.31), "常州": (31.78, 119.95),
    "中山": (22.52, 113.39), "珠海": (22.27, 113.57),
    "惠州": (23.08, 114.42), "江门": (22.58, 113.08),
    "汕头": (23.35, 116.68), "湛江": (21.27, 110.36),
}
