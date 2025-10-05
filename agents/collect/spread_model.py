from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .model_dataset import FEATURE_COLUMNS, load_training_dataset
from agents.storage import write_json_blob, read_json_blob


MODEL_BLOB_PATH = "models/spread_model.json"


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
    df = df.copy()
    df = df.loc[df["market_spread_book"].notna()].copy()
    if df.empty:
        return df
    target = df["market_spread_book"].astype(float)

    feature_cols: List[str] = []
    for base in FEATURE_COLUMNS:
        home_col = f"{base}_home"
        away_col = f"{base}_away"
        if home_col in df.columns and away_col in df.columns:
            feature_name = f"delta_{base}"
            df[feature_name] = df[home_col].astype(float).fillna(0.0) - df[away_col].astype(float).fillna(0.0)
            feature_cols.append(feature_name)

    df = df[feature_cols]
    df["target"] = target
    df = df.dropna(subset=["target"])
    return df


def train_linear_model() -> LinearSpreadModel | None:
    df = load_training_dataset()
    df = _prepare_training_frame(df)
    if df.empty:
        return None

    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)

    if X.shape[0] < X.shape[1] + 1:
        return None

    X_ext = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(X_ext, y, rcond=None)
    intercept = float(beta[0])
    coefs = beta[1:]
    feature_names = df.drop(columns=["target"]).columns
    coefficients = {name: float(value) for name, value in zip(feature_names, coefs)}

    model = LinearSpreadModel(intercept=intercept, coefficients=coefficients)
    write_json_blob(MODEL_BLOB_PATH, {
        "intercept": intercept,
        "coefficients": coefficients,
        "features": list(feature_names),
    })
    return model


def load_linear_model() -> LinearSpreadModel | None:
    data = read_json_blob(MODEL_BLOB_PATH)
    if not data:
        return None
    try:
        return LinearSpreadModel(
            intercept=float(data["intercept"]),
            coefficients={k: float(v) for k, v in data.get("coefficients", {}).items()},
        )
    except Exception:
        return None


__all__ = ["train_linear_model", "load_linear_model", "LinearSpreadModel"]
