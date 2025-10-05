import pandas as pd

from agents.collect.model_dataset import load_training_dataset
from agents.collect import CfbdClients, ApiCache
from agents.collect.team_inputs import build_team_inputs_datadriven
from agents.collect.predictions import build_predictions_for_year


def test_load_training_dataset(tmp_path, monkeypatch):
    apis = CfbdClients(bearer_token="")
    cache = ApiCache(root=str(tmp_path / "cache"))

    team_inputs = build_team_inputs_datadriven(2024, apis, cache)
    team_inputs["season"] = 2024
    team_inputs.to_csv(tmp_path / "team_inputs.csv", index=False)

    preds = pd.DataFrame(
        {
            "week": [1],
            "date": ["2024-08-24"],
            "home_team": [team_inputs["team"].iloc[0]],
            "away_team": [team_inputs["team"].iloc[1]],
            "market_spread_book": [-3.5],
            "model_spread_book": [-4.0],
            "home_points": [24],
            "away_points": [21],
        }
    )

    monkeypatch.setattr("agents.collect.model_dataset.read_dataset", lambda name: preds if name == "upa_predictions" else team_inputs)

    dataset = load_training_dataset()
    assert not dataset.frame.empty
    assert "market_spread_book" in dataset.frame.columns
    assert dataset.feature_columns
    assert any(col.startswith("delta_") for col in dataset.feature_columns)
