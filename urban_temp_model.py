#!/usr/bin/env python3
"""
Urban Surface Temperature Regression
------------------------------------
Trains an XGBoost regressor on tabular climate + geospatial features.

Usage:
    python urban_temp_model.py --data data.csv --target surface_temp_c
    python urban_temp_model.py            # runs on synthetic demo data

Outputs:
    models/urban_temp_model.joblib   (full pipeline: scaler + model + metadata)
    models/metrics.json              (CV + hold-out metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
DEFAULT_FEATURES = [
    "green_space_ratio",
    "building_density",
    "wind_speed",
    "albedo",
    "air_temperature",
    "humidity",
    "elevation",
    "distance_to_water",
    "sky_view_factor",
]
DEFAULT_TARGET = "surface_temp_c"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("urban_temp")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_synthetic_data(n: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Physically-plausible synthetic data for demo/testing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "green_space_ratio": rng.beta(2, 5, n),
            "building_density": rng.beta(3, 3, n),
            "wind_speed": rng.gamma(2.0, 1.5, n),
            "albedo": rng.uniform(0.05, 0.45, n),
            "air_temperature": rng.normal(30, 4, n),
            "humidity": rng.uniform(20, 90, n),
            "elevation": rng.uniform(0, 500, n),
            "distance_to_water": rng.exponential(1500, n),
            "sky_view_factor": rng.uniform(0.2, 1.0, n),
        }
    )
    df[DEFAULT_TARGET] = (
        df["air_temperature"]
        - 6.0 * df["green_space_ratio"]
        + 5.0 * df["building_density"]
        - 0.5 * df["wind_speed"]
        - 8.0 * df["albedo"]
        - 0.004 * df["elevation"]
        - 0.0005 * np.minimum(df["distance_to_water"], 3000)
        + 2.0 * df["sky_view_factor"]
        + 0.8 * df["building_density"] * df["air_temperature"] / 30
        + rng.normal(0, 0.8, n)
    )
    return df


def load_data(path: str | None, target: str) -> pd.DataFrame:
    if path is None:
        log.warning("No --data supplied; using synthetic demo data.")
        return make_synthetic_data()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {list(df.columns)}")
    return df


def validate_and_clean(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[features + [target]].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna(subset=[target])  # never impute the label
    log.info("Dropped %d rows with missing target.", n_before - len(df))

    # XGBoost handles NaN in features natively, so we keep them as-is.
    n_nan = int(df[features].isna().sum().sum())
    if n_nan:
        log.info("%d missing feature values left for XGBoost to handle natively.", n_nan)

    if len(df) < 50:
        raise ValueError(f"Too few samples after cleaning: {len(df)}")
    return df


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_pipeline(features: list[str]) -> Pipeline:
    # Scaler lives INSIDE the pipeline so it is fit only on training folds
    # (prevents leakage in cross-validation).
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_cross_validation(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int) -> dict:
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,  # XGBoost already parallelises internally
        return_train_score=False,
    )
    out = {
        "cv_rmse_mean": float(-res["test_rmse"].mean()),
        "cv_rmse_std": float(res["test_rmse"].std()),
        "cv_r2_mean": float(res["test_r2"].mean()),
        "cv_r2_std": float(res["test_r2"].std()),
    }
    log.info(
        "%d-fold CV  RMSE: %.4f ± %.4f | R²: %.4f ± %.4f",
        folds, out["cv_rmse_mean"], out["cv_rmse_std"], out["cv_r2_mean"], out["cv_r2_std"],
    )
    return out


def feature_importance_report(
    pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, features: list[str]
) -> pd.DataFrame:
    model: XGBRegressor = pipe.named_steps["model"]
    gain = model.get_booster().get_score(importance_type="gain")
    # Booster uses f0, f1, ... names when fed numpy arrays from the ColumnTransformer
    gain_vec = np.array([gain.get(f"f{i}", 0.0) for i in range(len(features))])
    gain_vec = gain_vec / gain_vec.sum() if gain_vec.sum() > 0 else gain_vec

    perm = permutation_importance(
        pipe, X_test, y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
    )
    report = (
        pd.DataFrame(
            {
                "feature": features,
                "gain_importance": gain_vec,
                "perm_importance_rmse": perm.importances_mean,
                "perm_importance_std": perm.importances_std,
            }
        )
        .sort_values("perm_importance_rmse", ascending=False)
        .reset_index(drop=True)
    )
    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train urban surface temperature model.")
    ap.add_argument("--data", type=str, default=None, help="CSV/Parquet path (default: synthetic).")
    ap.add_argument("--target", type=str, default=DEFAULT_TARGET)
    ap.add_argument("--features", nargs="+", default=None, help="Feature columns.")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default="models")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    features = args.features or DEFAULT_FEATURES

    df = load_data(args.data, args.target)
    df = validate_and_clean(df, features, args.target)
    log.info("Dataset: %d rows, %d features.", len(df), len(features))

    X, y = df[features], df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )
    log.info("Train: %d | Test: %d", len(X_train), len(X_test))

    pipe = build_pipeline(features)

    # 1) Cross-validation on training data only
    cv_metrics = run_cross_validation(pipe, X_train, y_train, args.cv_folds)

    # 2) Final fit + hold-out evaluation
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    test_metrics = {"test_rmse": rmse(y_test, preds), "test_r2": float(r2_score(y_test, preds))}
    log.info("Hold-out  RMSE: %.4f | R²: %.4f", test_metrics["test_rmse"], test_metrics["test_r2"])

    # 3) Feature importances
    report = feature_importance_report(pipe, X_test, y_test, features)
    print("\n=== Feature Importances ===")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 4) Persist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "urban_temp_model.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "features": features,
            "target": args.target,
            "metrics": {**cv_metrics, **test_metrics},
        },
        model_path,
        compress=3,
    )
    (out_dir / "metrics.json").write_text(json.dumps({**cv_metrics, **test_metrics}, indent=2))
    log.info("Saved model to %s", model_path)

    # Example inference:
    #   bundle = joblib.load("models/urban_temp_model.joblib")
    #   bundle["pipeline"].predict(new_df[bundle["features"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
