import importlib.util
from pathlib import Path


APP_PATH = Path(__file__).with_name("app.py")


def load_demo_app():
    spec = importlib.util.spec_from_file_location("demo_app_for_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_models_are_available_and_loadable():
    app = load_demo_app()

    available = app.get_available_models()

    assert "LSTM (Legacy)" in available
    assert "TSMixer (Legacy)" in available

    lstm, lstm_loaded, _, _ = app.load_model({}, "LSTM (Legacy)")
    tsmixer, tsmixer_loaded, _, _ = app.load_model({}, "TSMixer (Legacy)")

    assert lstm is not None
    assert tsmixer is not None
    assert lstm_loaded
    assert tsmixer_loaded


def test_future_dates_are_after_last_market_date():
    app = load_demo_app()

    future_dates = app.build_future_dates("2026-04-24", 5)

    assert len(future_dates) == 5
    assert all(str(day.date()) > "2026-04-24" for day in future_dates)
    assert all(day.dayofweek < 5 for day in future_dates)


def test_price_rmse_band_wraps_forecast_path():
    app = load_demo_app()

    predicted = [101.0, 102.0, 103.0]
    lower, upper = app.build_price_rmse_band(100.0, predicted, [0.01, 0.02, 0.03])

    assert len(lower) == len(predicted)
    assert len(upper) == len(predicted)
    assert all(lo < mid < hi for lo, mid, hi in zip(lower, predicted, upper))


def test_rmse_fill_series_includes_latest_close_anchor():
    app = load_demo_app()

    dates, lower, upper = app.build_rmse_fill_series(
        "2026-04-24",
        100.0,
        app.build_future_dates("2026-04-24", 2),
        [101.0, 102.0],
        [0.01, 0.02],
    )

    assert len(dates) == 3
    assert dates[0].strftime("%Y-%m-%d") == "2026-04-24"
    assert lower[0] == 100.0
    assert upper[0] == 100.0
