from __future__ import annotations

import os

from agents.collect import CfbdClients, ApiCache
from agents.collect.model_dataset import load_training_dataset
from agents.collect.spread_model import train_linear_model


def main() -> None:
    # Ensure dataset exists (training uses current DB, but the job can be run standalone)
    apis = CfbdClients(bearer_token=os.environ.get("CFBD_BEARER_TOKEN", ""))
    cache = ApiCache()
    _ = load_training_dataset()

    model = train_linear_model()
    if model is None:
        print("[warn] train_spread_model: insufficient data to train")
    else:
        print(
            "[ok] train_spread_model: trained linear model with",
            f"{len(model.coefficients)} features",
        )


if __name__ == "__main__":
    main()
