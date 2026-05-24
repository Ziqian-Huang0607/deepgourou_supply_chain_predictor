"""models.py - LightGBM, XGBoost, Ridge forecaster models."""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from config import LGBM_PARAMS, LGBM_ROUNDS, LGBM_EARLY, XGB_PARAMS, XGB_ROUNDS, XGB_EARLY, RIDGE_ALPHA
from utils import get_logger

logger = get_logger("models")


class LightGBMForecaster:
    def __init__(self):
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_valid=None, y_valid=None):
        import lightgbm as lgb
        dtrain = lgb.Dataset(X, label=y)
        valid_sets, callbacks = [dtrain], []
        if X_valid is not None and y_valid is not None:
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
            valid_sets.append(dvalid)
            callbacks = [lgb.early_stopping(LGBM_EARLY, verbose=False),
                         lgb.log_evaluation(period=0)]
        self.model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=LGBM_ROUNDS,
                               valid_sets=valid_sets, callbacks=callbacks)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def feature_importance(self) -> list:
        imp = self.model.feature_importance(importance_type="gain")
        names = self.model.feature_name()
        return sorted(zip(names, imp), key=lambda x: x[1], reverse=True)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model.model_to_string(), f)

    @classmethod
    def load(cls, path: str):
        import lightgbm as lgb
        inst = cls()
        with open(path, "rb") as f:
            inst.model = lgb.Booster(model_str=pickle.load(f))
        return inst


class XGBoostForecaster:
    def __init__(self):
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_valid=None, y_valid=None):
        from xgboost import DMatrix, train
        dtrain = DMatrix(X, label=y)
        evals = [(dtrain, "train")]
        if X_valid is not None and y_valid is not None:
            dvalid = DMatrix(X_valid, label=y_valid)
            evals.append((dvalid, "valid"))
        self.model = train(XGB_PARAMS, dtrain, num_boost_round=XGB_ROUNDS,
                           evals=evals, early_stopping_rounds=XGB_EARLY, verbose_eval=False)
        return self

    def predict(self, X) -> np.ndarray:
        from xgboost import DMatrix
        if hasattr(X, 'columns'):
            return self.model.predict(DMatrix(X), iteration_range=(0, self.model.best_iteration + 1))
        else:
            fnames = self.model.feature_names
            X_df = pd.DataFrame(X, columns=fnames)
            return self.model.predict(DMatrix(X_df), iteration_range=(0, self.model.best_iteration + 1))

    def save(self, path: str):
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str):
        from xgboost import Booster
        inst = cls()
        inst.model = Booster()
        inst.model.load_model(path)
        return inst


class RidgeForecaster:
    def __init__(self):
        self.model = Ridge(alpha=RIDGE_ALPHA)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            inst = cls()
            inst.model = pickle.load(f)
            return inst