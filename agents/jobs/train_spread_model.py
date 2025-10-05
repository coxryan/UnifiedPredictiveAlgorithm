from __future__ import annotations

import os

from agents.collect import CfbdClients, ApiCache
from agents.collect.model_dataset import load_training_dataset
from agents.collect.spread_model import train_residual_model


def main() -> None:
    # Ensure dataset exists (training uses current DB, but the job can be run standalone)
    apis = CfbdClients(bearer_token=os.environ.get("CFBD_BEARER_TOKEN", ""))
    cache = ApiCache()
    dataset = load_training_dataset()
    if dataset.frame.empty or not dataset.feature_columns:
        print("[warn] train_spread_model: dataset empty; skipping training")
        return

    model = train_residual_model()
    if model is None:
        print("[warn] train_spread_model: insufficient data to train residual model")
    else:
        metrics = model.metrics.get("training", {})
        mae_base = metrics.get("mae_baseline")
        mae_model = metrics.get("mae_calibrated")
        improvement = None
        if mae_base and mae_model is not None:
            improvement = (1.0 - mae_model / mae_base) * 100.0 if mae_base else None
        print("[ok] train_spread_model: trained residual model")
        if mae_base is not None:
            print(f"  - baseline MAE: {mae_base:.3f}")
        if mae_model is not None:
            print(f"  - calibrated MAE: {mae_model:.3f}")
        if improvement is not None:
            print(f"  - MAE improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()
