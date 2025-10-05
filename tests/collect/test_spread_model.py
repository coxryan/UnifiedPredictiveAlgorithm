import pandas as pd

from agents.collect.spread_model import _prepare_training_frame, LinearSpreadModel


def test_prepare_training_frame_creates_deltas():
    df = pd.DataFrame(
        {
            "market_spread_book": [-3.5, -7.0],
            "stat_off_index_0_100_home": [70, 65],
            "stat_off_index_0_100_away": [60, 55],
            "stat_def_index_0_100_home": [55, 50],
            "stat_def_index_0_100_away": [45, 40],
        }
    )
    frame = _prepare_training_frame(df)
    assert "delta_stat_off_index_0_100" in frame.columns
    assert "delta_stat_def_index_0_100" in frame.columns
    assert frame.shape[0] == 2


def test_linear_model_predict():
    model = LinearSpreadModel(intercept=1.0, coefficients={"delta_stat_off_index_0_100": 0.1})
    pred = model.predict({"delta_stat_off_index_0_100": 10})
    assert abs(pred - 2.0) < 1e-6
