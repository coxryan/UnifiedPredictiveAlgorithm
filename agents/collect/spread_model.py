from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .model_dataset import TrainingDataset, load_training_dataset
from agents.storage import read_json_blob, write_json_blob

MODEL_BLOB_PATH = "models/residual_model.json"


@dataclass
class RidgeModel:
    alpha: float
    intercept: float
    coefficients: Dict[str, float]

    def predict(self, features: Dict[str, float]) -> float:
        value = self.intercept
        for key, coef in self.coefficients.items():
            value += coef * float(features.get(key, 0.0))
        return value


@dataclass
class LinearSpreadModel:
    intercept: float
    coefficients: Dict[str, float]

    def predict(self, features: Dict[str, float]) -> float:
        value = self.intercept
        for key, coef in self.coefficients.items():
            value += coef * float(features.get(key, 0.0))
        return value


def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper kept for backward compatibility with older tests.

    Produces home/away deltas and a simple residual target even when only
    partial columns are provided.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    frame = df.copy()
    if "market_spread_book" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["market_spread_book"], errors="coerce").notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    delta_columns: List[str] = []
    for col in list(frame.columns):
        if not col.endswith("_home"):
            continue
        base = col[:-5]
        away_col = f"{base}_away"
        if away_col not in frame.columns:
            continue
        home_series = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        away_series = pd.to_numeric(frame[away_col], errors="coerce").fillna(0.0)
        delta_name = f"delta_{base}"
        frame[delta_name] = home_series - away_series
        delta_columns.append(delta_name)

    market = pd.to_numeric(frame.get("market_spread_book"), errors="coerce").fillna(0.0)
    actual_raw = frame.get("actual_margin")
    if isinstance(actual_raw, pd.Series):
        actual = pd.to_numeric(actual_raw, errors="coerce").fillna(0.0)
    else:
        actual = pd.Series(0.0, index=frame.index)
    frame["target"] = actual - market

    keep = delta_columns + ["target"]
    return frame[keep]


@dataclass
class DecisionStump:
    feature: str
    threshold: float
    left_value: float
    right_value: float

    def predict(self, x: float) -> float:
        return self.left_value if x <= self.threshold else self.right_value


@dataclass
class GradientBoostingModel:
    learning_rate: float
    stumps: List[DecisionStump] = field(default_factory=list)

    def predict_matrix(self, X: np.ndarray, feature_order: Sequence[str]) -> np.ndarray:
        if not self.stumps:
            return np.zeros(len(X), dtype=float)
        predictions = np.zeros(len(X), dtype=float)
        feature_index = {name: idx for idx, name in enumerate(feature_order)}
        for stump in self.stumps:
            idx = feature_index.get(stump.feature)
            if idx is None:
                continue
            column = X[:, idx]
            mask = column <= stump.threshold
            predictions[mask] += self.learning_rate * stump.left_value
            predictions[~mask] += self.learning_rate * stump.right_value
        return predictions


@dataclass
class Calibration:
    bias: float
    scale: float

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return self.bias + self.scale * raw


@dataclass
class ResidualEnsembleModel:
    features: Sequence[str]
    ridge: RidgeModel
    gbdt: Optional[GradientBoostingModel]
    calibration: Calibration
    metrics: Dict[str, Any]
    residual_std: float
    norm_stats: Dict[str, Any]

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return _apply_norm_stats(frame, self.features, self.norm_stats)

    def predict_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        ridge_pred = _ridge_predict_matrix(self.ridge, X, self.features)
        gbdt_pred = (
            self.gbdt.predict_matrix(X, self.features) if self.gbdt else np.zeros_like(ridge_pred)
        )
        raw = ridge_pred + gbdt_pred
        calibrated = self.calibration.apply(raw)
        return {
            "ridge": ridge_pred,
            "gbdt": gbdt_pred,
            "raw": raw,
            "calibrated": calibrated,
        }

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.predict_components(X)["calibrated"]

    def predict_adjustment(self, features: Dict[str, float], season: Optional[int] = None) -> float:
        row = {name: float(features.get(name, 0.0)) for name in self.features}
        frame = pd.DataFrame([row])
        if season is not None:
            frame["season"] = season
        normalized = self.transform_features(frame)
        x = normalized[self.features].to_numpy(dtype=float)
        return float(self.predict_matrix(x)[0])


def _ridge_predict_matrix(model: RidgeModel, X: np.ndarray, feature_order: Sequence[str]) -> np.ndarray:
    coef_vector = np.array([model.coefficients.get(name, 0.0) for name in feature_order], dtype=float)
    return model.intercept + X @ coef_vector


def _ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    feature_names: Sequence[str],
    eps: float = 1e-8,
) -> RidgeModel:
    n_features = X.shape[1]
    identity = np.eye(n_features + 1)
    identity[0, 0] = 0.0  # do not regularize intercept
    X_ext = np.hstack([np.ones((X.shape[0], 1)), X])
    ridge_matrix = X_ext.T @ X_ext + alpha * identity
    rhs = X_ext.T @ y
    beta = np.linalg.solve(ridge_matrix + eps * np.eye(n_features + 1), rhs)
    intercept = float(beta[0])
    coefficients = beta[1:]
    return RidgeModel(
        alpha=alpha,
        intercept=intercept,
        coefficients={name: float(value) for name, value in zip(feature_names, coefficients)},
    )


def _candidate_thresholds(values: np.ndarray, max_bins: int = 10) -> np.ndarray:
    quantiles = np.linspace(0.1, 0.9, max_bins)
    candidates = np.unique(np.quantile(values, quantiles))
    return candidates


def _best_stump(X: np.ndarray, residuals: np.ndarray, feature_names: Sequence[str]) -> Optional[DecisionStump]:
    best_feature: Optional[str] = None
    best_threshold: float = 0.0
    best_loss: float = math.inf
    best_left: float = 0.0
    best_right: float = 0.0
    for idx, name in enumerate(feature_names):
        column = X[:, idx]
        if np.allclose(column, column[0]):
            continue
        thresholds = _candidate_thresholds(column)
        for threshold in thresholds:
            left_mask = column <= threshold
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            left_val = residuals[left_mask].mean()
            right_val = residuals[right_mask].mean()
            loss_left = ((residuals[left_mask] - left_val) ** 2).sum()
            loss_right = ((residuals[right_mask] - right_val) ** 2).sum()
            loss = loss_left + loss_right
            if loss < best_loss:
                best_loss = loss
                best_feature = name
                best_threshold = float(threshold)
                best_left = float(left_val)
                best_right = float(right_val)
    if best_feature is None:
        return None
    return DecisionStump(best_feature, best_threshold, best_left, best_right)


def _fit_gbdt(
    X: np.ndarray,
    residuals: np.ndarray,
    feature_names: Sequence[str],
    *,
    learning_rate: float,
    n_estimators: int,
) -> GradientBoostingModel:
    model = GradientBoostingModel(learning_rate=learning_rate)
    current_residuals = residuals.copy()
    for _ in range(n_estimators):
        stump = _best_stump(X, current_residuals, feature_names)
        if stump is None:
            break
        model.stumps.append(stump)
        idx = feature_names.index(stump.feature)
        column = X[:, idx]
        mask = column <= stump.threshold
        update = np.where(mask, stump.left_value, stump.right_value)
        current_residuals -= learning_rate * update
    return model


def _calibrate(y_true: np.ndarray, y_pred: np.ndarray) -> Calibration:
    if len(y_pred) == 0:
        return Calibration(bias=0.0, scale=1.0)
    X = np.vstack([np.ones_like(y_pred), y_pred]).T
    beta, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    bias = float(beta[0])
    scale = float(beta[1])
    if math.isnan(scale) or math.isinf(scale):
        scale = 1.0
    if math.isnan(bias) or math.isinf(bias):
        bias = 0.0
    return Calibration(bias=bias, scale=scale)


def _baseline_mae(y: np.ndarray) -> float:
    return float(np.mean(np.abs(y))) if len(y) else math.nan


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else math.nan


def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _coverage(y_true: np.ndarray, y_pred: np.ndarray, window: float) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(np.mean(np.abs(y_true - y_pred) <= window))


def _compute_norm_stats(frame: pd.DataFrame, features: Sequence[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if frame.empty:
        zeros = {feat: 0.0 for feat in features}
        ones = {feat: 1.0 for feat in features}
        stats["__global__"] = {"mean": zeros, "std": ones}
        return stats

    global_mean = frame[features].mean().to_dict()
    global_std = (
        frame[features].std().replace(0.0, np.nan).fillna(1.0).to_dict()
    )
    stats["__global__"] = {"mean": global_mean, "std": global_std}

    if "season" in frame.columns:
        season_series = frame["season"]
        for season in sorted(season_series.dropna().unique()):
            mask = season_series == season
            subset = frame.loc[mask, features]
            if subset.empty:
                continue
            stats[str(int(season))] = {
                "mean": subset.mean().to_dict(),
                "std": subset.std().replace(0.0, np.nan).fillna(1.0).to_dict(),
            }
        if season_series.isna().any():
            subset = frame.loc[season_series.isna(), features]
            if not subset.empty:
                stats["unknown"] = {
                    "mean": subset.mean().to_dict(),
                    "std": subset.std().replace(0.0, np.nan).fillna(1.0).to_dict(),
                }
    return stats


def _apply_norm_stats(
    frame: pd.DataFrame,
    features: Sequence[str],
    stats: Dict[str, Any],
) -> pd.DataFrame:
    for feat in features:
        if feat not in frame.columns:
            frame[feat] = 0.0
    normalized = pd.DataFrame(index=frame.index, columns=features, dtype=float)
    global_params = stats.get("__global__", {"mean": {}, "std": {}})
    global_mean = pd.Series(global_params.get("mean", {}))
    global_std = pd.Series(global_params.get("std", {})).replace(0.0, np.nan).fillna(1.0)

    if "season" in frame.columns:
        grouping = frame["season"].fillna("unknown")
    else:
        grouping = pd.Series(["unknown"] * len(frame), index=frame.index)

    for key, indices in grouping.groupby(grouping).groups.items():
        params = None
        if key == "unknown":
            params = stats.get("unknown")
        else:
            try:
                params = stats.get(str(int(key)))
            except (TypeError, ValueError):
                params = stats.get(str(key))
        if params is None:
            params = {"mean": global_mean.to_dict(), "std": global_std.to_dict()}

        mean_series = pd.Series(params.get("mean", {})).reindex(features).fillna(0.0)
        std_series = (
            pd.Series(params.get("std", {})).reindex(features).replace(0.0, np.nan).fillna(1.0)
        )
        subset = frame.loc[indices, features]
        normalized.loc[indices, features] = (subset - mean_series) / std_series

    return normalized.fillna(0.0)


def _group_folds(groups: Sequence[Any]) -> List[np.ndarray]:
    indices: Dict[Any, List[int]] = {}
    for idx, group in enumerate(groups):
        indices.setdefault(group, []).append(idx)
    group_indices = [np.array(v, dtype=int) for v in indices.values() if v]
    if len(group_indices) < 2:
        return []
    return group_indices


def _cross_validate(
    dataset: TrainingDataset,
    alphas: Sequence[float],
    *,
    learning_rate: float,
    n_estimators: int,
) -> Dict[str, Any]:
    frame = dataset.frame
    features = list(dataset.feature_columns)
    if frame.empty or not features:
        return {}

    season_groups = frame["season"].to_numpy()
    folds = _group_folds(season_groups)
    if not folds:
        week_groups = frame["cv_key"].to_numpy()
        folds = _group_folds(week_groups)
    if not folds:
        return {}

    results: Dict[float, Dict[str, float]] = {}
    for alpha in alphas:
        fold_metrics: List[Dict[str, float]] = []
        for val_idx in folds:
            train_mask = np.ones(len(frame), dtype=bool)
            train_mask[val_idx] = False
            if train_mask.sum() <= len(features) + 1:
                continue

            train_frame = frame.iloc[train_mask]
            val_frame = frame.iloc[val_idx]

            y_train = train_frame[dataset.target_column].to_numpy(dtype=float)
            y_val = val_frame[dataset.target_column].to_numpy(dtype=float)

            norm_stats = _compute_norm_stats(train_frame, features)
            X_train_df = _apply_norm_stats(train_frame, features, norm_stats)
            X_val_df = _apply_norm_stats(val_frame, features, norm_stats)

            X_train = X_train_df[features].to_numpy(dtype=float)
            X_val = X_val_df[features].to_numpy(dtype=float)

            ridge = _ridge_fit(X_train, y_train, alpha, features)
            ridge_train_pred = _ridge_predict_matrix(ridge, X_train, features)
            ridge_val_pred = _ridge_predict_matrix(ridge, X_val, features)
            residual_train = y_train - ridge_train_pred

            gbdt = _fit_gbdt(
                X_train,
                residual_train,
                features,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
            )
            gbdt_val_pred = gbdt.predict_matrix(X_val, features)
            ensemble_val = ridge_val_pred + gbdt_val_pred

            baseline_mae = _baseline_mae(y_val)
            ridge_mae = _mae(y_val, ridge_val_pred)
            ensemble_mae = _mae(y_val, ensemble_val)
            fold_metrics.append(
                {
                    "fold_size": float(len(val_idx)),
                    "mae_baseline": baseline_mae,
                    "mae_ridge": ridge_mae,
                    "mae_ensemble": ensemble_mae,
                    "sign_accuracy": _sign_accuracy(y_val, ensemble_val),
                    "coverage_3": _coverage(y_val, ensemble_val, 3.0),
                    "coverage_7": _coverage(y_val, ensemble_val, 7.0),
                }
            )
        if fold_metrics:
            total = sum(item["fold_size"] for item in fold_metrics)
            def weighted_mean(key: str) -> float:
                return float(
                    sum(item[key] * item["fold_size"] for item in fold_metrics) / total
                )

            results[alpha] = {
                "mae_baseline": weighted_mean("mae_baseline"),
                "mae_ridge": weighted_mean("mae_ridge"),
                "mae_ensemble": weighted_mean("mae_ensemble"),
                "mae_improvement_pct": (
                    (1.0 - weighted_mean("mae_ensemble") / weighted_mean("mae_baseline")) * 100.0
                    if weighted_mean("mae_baseline")
                    else math.nan
                ),
                "sign_accuracy": weighted_mean("sign_accuracy"),
                "coverage_3": weighted_mean("coverage_3"),
                "coverage_7": weighted_mean("coverage_7"),
                "folds": fold_metrics,
            }
    return results


def _select_alpha(cv_results: Dict[float, Dict[str, float]]) -> float:
    if not cv_results:
        return 1.5
    ranked = sorted(
        cv_results.items(),
        key=lambda item: (
            item[1].get("mae_ensemble", math.inf)
            if not math.isnan(item[1].get("mae_ensemble", math.inf))
            else math.inf
        ),
    )
    return float(ranked[0][0])


def train_residual_model(
    *,
    alpha_grid: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    learning_rate: float = 0.3,
    n_estimators: int = 50,
) -> Optional[ResidualEnsembleModel]:
    dataset = load_training_dataset()
    frame = dataset.frame
    features = list(dataset.feature_columns)
    if frame.empty or not features:
        return None

    cv_results = _cross_validate(dataset, alpha_grid, learning_rate=learning_rate, n_estimators=n_estimators)
    alpha = _select_alpha(cv_results)

    norm_stats = _compute_norm_stats(frame, features)
    normalized_frame = _apply_norm_stats(frame, features, norm_stats)
    X = normalized_frame[features].to_numpy(dtype=float)
    y = frame[dataset.target_column].to_numpy(dtype=float)

    ridge = _ridge_fit(X, y, alpha, features)
    ridge_pred = _ridge_predict_matrix(ridge, X, features)
    residuals = y - ridge_pred

    gbdt = _fit_gbdt(
        X,
        residuals,
        features,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
    )
    gbdt_pred = gbdt.predict_matrix(X, features)
    ensemble_pred = ridge_pred + gbdt_pred

    calibration = _calibrate(y, ensemble_pred)
    calibrated_pred = calibration.apply(ensemble_pred)
    residual_std = float(np.std(y - calibrated_pred)) if len(y) else 0.0

    metrics = {
        "alpha": alpha,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "training": {
            "mae_baseline": _baseline_mae(y),
            "mae_ridge": _mae(y, ridge_pred),
            "mae_ensemble": _mae(y, ensemble_pred),
            "mae_calibrated": _mae(y, calibrated_pred),
            "sign_accuracy": _sign_accuracy(y, calibrated_pred),
            "coverage_3": _coverage(y, calibrated_pred, 3.0),
            "coverage_7": _coverage(y, calibrated_pred, 7.0),
        },
        "cv": cv_results,
    }

    model = ResidualEnsembleModel(
        features=features,
        ridge=ridge,
        gbdt=gbdt if gbdt.stumps else None,
        calibration=calibration,
        metrics=metrics,
        residual_std=residual_std,
        norm_stats=norm_stats,
    )

    write_json_blob(MODEL_BLOB_PATH, serialize_model(model))
    return model


def serialize_model(model: ResidualEnsembleModel) -> Dict[str, Any]:
    payload = {
        "version": 1,
        "features": list(model.features),
        "ridge": {
            "alpha": model.ridge.alpha,
            "intercept": model.ridge.intercept,
            "coefficients": model.ridge.coefficients,
        },
        "calibration": {
            "bias": model.calibration.bias,
            "scale": model.calibration.scale,
        },
        "metrics": model.metrics,
        "residual_std": model.residual_std,
        "norm_stats": model.norm_stats,
    }
    if model.gbdt and model.gbdt.stumps:
        payload["gbdt"] = {
            "learning_rate": model.gbdt.learning_rate,
            "stumps": [
                {
                    "feature": stump.feature,
                    "threshold": stump.threshold,
                    "left_value": stump.left_value,
                    "right_value": stump.right_value,
                }
                for stump in model.gbdt.stumps
            ],
        }
    return payload


def deserialize_model(payload: Dict[str, Any]) -> ResidualEnsembleModel:
    features = payload.get("features", [])
    ridge_data = payload.get("ridge", {})
    ridge = RidgeModel(
        alpha=float(ridge_data.get("alpha", 1.5)),
        intercept=float(ridge_data.get("intercept", 0.0)),
        coefficients={k: float(v) for k, v in ridge_data.get("coefficients", {}).items()},
    )
    gbdt_payload = payload.get("gbdt")
    gbdt = None
    if gbdt_payload:
        stumps = [
            DecisionStump(
                feature=stump.get("feature", ""),
                threshold=float(stump.get("threshold", 0.0)),
                left_value=float(stump.get("left_value", 0.0)),
                right_value=float(stump.get("right_value", 0.0)),
            )
            for stump in gbdt_payload.get("stumps", [])
        ]
        gbdt = GradientBoostingModel(
            learning_rate=float(gbdt_payload.get("learning_rate", 0.3)),
            stumps=stumps,
        )
    calibration_payload = payload.get("calibration", {})
    calibration = Calibration(
        bias=float(calibration_payload.get("bias", 0.0)),
        scale=float(calibration_payload.get("scale", 1.0)),
    )
    metrics = payload.get("metrics", {})
    residual_std = float(payload.get("residual_std", 1.0))
    norm_stats = payload.get("norm_stats", {})
    return ResidualEnsembleModel(
        features=features,
        ridge=ridge,
        gbdt=gbdt,
        calibration=calibration,
        metrics=metrics,
        residual_std=residual_std,
        norm_stats=norm_stats,
    )


def load_residual_model() -> Optional[ResidualEnsembleModel]:
    payload = read_json_blob(MODEL_BLOB_PATH)
    if not payload:
        return None
    try:
        return deserialize_model(payload)
    except Exception:
        return None


__all__ = [
    "ResidualEnsembleModel",
    "LinearSpreadModel",
    "_prepare_training_frame",
    "train_residual_model",
    "load_residual_model",
    "serialize_model",
]
