import importlib.util
from pathlib import Path
import pandas as pd


APP_PATH = Path(__file__).with_name("app.py")


def load_demo_app():
    spec = importlib.util.spec_from_file_location("demo_app_for_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_models_are_available_and_loadable():
    app = load_demo_app()

    available = app.get_available_models()

    assert "LSTM" in available
    assert "TSMixer" in available

    lstm, lstm_loaded, _, _ = app.load_model({}, "LSTM")
    tsmixer, tsmixer_loaded, _, _ = app.load_model({}, "TSMixer")

    assert lstm is not None
    assert tsmixer is not None
    assert lstm_loaded
    assert tsmixer_loaded


def test_tft_models_are_available_and_loadable():
    app = load_demo_app()

    available = app.get_available_models()

    assert "TFT" in available
    assert "TFT-FinBERT" in available

    base_config = {
        "model": {
            "input_size": 12,
            "output_size": 5,
            "hidden_size": 128,
            "num_heads": 4,
            "dropout": 0.1,
            "lstm_layers": 1,
        },
        "num_articles": 4,
        "max_window_size": 14,
        "time_series_features": 12,
        "local_files_only": False,
    }

    standalone, standalone_loaded, _, standalone_path = app.load_model(base_config, "TFT")
    multimodal, multimodal_loaded, _, multimodal_path = app.load_model(base_config, "TFT-FinBERT")

    assert standalone is not None
    assert multimodal is not None
    assert standalone_loaded
    assert multimodal_loaded
    assert Path(standalone_path).parts[-2:] == ("experiments", "tft")
    assert Path(multimodal_path).parts[-2:] == ("experiments", "tft_finbert")


def test_backtest_window_uses_past_context_and_known_actuals():
    app = load_demo_app()
    df = pd.DataFrame({
        "date": pd.bdate_range("2026-04-01", periods=10),
        "close": range(100, 110),
    })

    valid_dates = app.get_valid_backtest_start_dates(df, seq_len=4, horizon=3)
    context, actual = app.prepare_backtest_window(df, seq_len=4, horizon=3, backtest_start_date=valid_dates[-1])

    assert [d.strftime("%Y-%m-%d") for d in valid_dates] == [
        "2026-04-07",
        "2026-04-08",
        "2026-04-09",
        "2026-04-10",
    ]
    assert context["close"].tolist() == [103, 104, 105, 106]
    assert actual["close"].tolist() == [107, 108, 109]


def test_find_date_position_handles_datetime_index():
    app = load_demo_app()
    dates = pd.DatetimeIndex(pd.bdate_range("2026-04-01", periods=5))

    assert app.find_date_position(dates, "2026-04-03") == 2


def test_latest_backtest_start_date_uses_most_recent_valid_window():
    app = load_demo_app()
    dates = pd.bdate_range("2026-04-01", periods=10)

    assert app.get_latest_backtest_start_date(list(dates)) == dates[-1]


def test_fetch_news_data_returns_blanks_when_yfinance_news_fails(monkeypatch):
    app = load_demo_app()

    class BrokenTicker:
        @property
        def news(self):
            raise ValueError("bad json")

    monkeypatch.setattr(app.yf, "Ticker", lambda ticker: BrokenTicker())
    app.fetch_news_data.clear()

    assert app.fetch_news_data("AAPL", max_articles=3) == ["", "", ""]


def test_backtest_metrics_compare_predicted_to_actual_and_direction():
    app = load_demo_app()

    metrics = app.compute_backtest_metrics(
        context_last_price=100.0,
        actual_prices=[101.0, 99.0, 103.0],
        predicted_prices=[102.0, 98.0, 104.0],
    )

    assert metrics["mae"] == 1.0
    assert round(metrics["rmse"], 4) == 1.0
    assert round(metrics["directional_accuracy"], 4) == 100.0


def test_backtest_figure_labels_actual_predicted_and_context_divider():
    app = load_demo_app()

    fig = app.build_backtest_figure(
        history_dates=list(pd.bdate_range("2026-04-01", periods=4)),
        history_prices=[100.0, 101.0, 102.0, 103.0],
        backtest_dates=list(pd.bdate_range("2026-04-07", periods=2)),
        actual_prices=[104.0, 105.0],
        pred_prices=[103.5, 105.5],
    )

    trace_names = [trace.name for trace in fig.data]
    assert "Actual" in trace_names
    assert "Predicted" in trace_names
    assert fig.layout.shapes[0].type == "line"
