import datetime
import importlib
import importlib.util
import json
import logging
import os
import pickle
import sys
import warnings
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import torch
import plotly.graph_objects as go
import yaml
from sklearn.exceptions import InconsistentVersionWarning

try:
    torch.classes.__path__ = []
except Exception:
    pass

yfinance_logger = logging.getLogger("yfinance")
yfinance_logger.setLevel(logging.CRITICAL)
yfinance_logger.propagate = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SRC_MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "models")
SRC_PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, "src", "preProcessing")
LSTM_LEGACY_DIR = os.path.join(PROJECT_ROOT, "cookbooks", "legacy", "lstm")
TSMIXER_LEGACY_DIR = os.path.join(PROJECT_ROOT, "cookbooks", "legacy", "tsmixer")
EXPERIMENT_TFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "tft_model.py")
EXPERIMENT_TFT_MULTIMODAL_PATH = os.path.join(
    PROJECT_ROOT, "src", "models", "TftMultiModalBaseline.py"
)

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("torchvision.transforms", None)

try:
    from transformers import AutoTokenizer
    import transformers.utils.import_utils as transformers_import_utils
    transformers_import_utils.is_torchvision_available = lambda: False
    transformers_import_utils.is_torchvision_v2_available = lambda: False
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

for path in (PROJECT_ROOT, SRC_MODELS_DIR):
    if path not in sys.path:
        sys.path.append(path)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

TFT_IMPORT_ERROR = None
MULTIMODAL_IMPORT_ERROR = None
LEGACY_IMPORT_ERROR = None

def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_module_from_repo_path(module_name, path, aliases=()):
    if not os.path.exists(path):
        raise ImportError(f"Missing module file: {path}")
    module = import_from_path(module_name, path)
    for alias in aliases:
        sys.modules[alias] = module
    return module


try:
    tft_model_module = import_module_from_repo_path(
        "demo_tft_model",
        EXPERIMENT_TFT_MODEL_PATH,
        aliases=("tft_model", "src.models.tft_model"),
    )
    TFTModel = tft_model_module.TFTModel
except ImportError as exc:
    TFTModel = None
    TFT_IMPORT_ERROR = exc

try:
    multimodal_module = import_module_from_repo_path(
        "demo_tft_multimodal",
        EXPERIMENT_TFT_MULTIMODAL_PATH,
        aliases=("TftMultiModalBaseline", "src.models.TftMultiModalBaseline"),
    )
    TftMultiModalStockPredictor = multimodal_module.MultiModalStockPredictor
except ImportError as exc:
    TftMultiModalStockPredictor = None
    MULTIMODAL_IMPORT_ERROR = exc


try:
    lstm_model_module = import_from_path(
        "legacy_lstm_model",
        os.path.join(LSTM_LEGACY_DIR, "src", "models", "lstm.py"),
    )
    lstm_features_module = import_from_path(
        "legacy_lstm_features",
        os.path.join(LSTM_LEGACY_DIR, "src", "preprocessing", "features.py"),
    )
    tsmixer_model_module = import_from_path(
        "legacy_tsmixer_model",
        os.path.join(TSMIXER_LEGACY_DIR, "src", "models", "tsmixer.py"),
    )
    tsmixer_features_module = import_from_path(
        "legacy_tsmixer_features",
        os.path.join(TSMIXER_LEGACY_DIR, "src", "preprocessing", "features.py"),
    )
    LSTMForecaster = lstm_model_module.LSTMForecaster
    build_lstm_features = lstm_features_module.build_features
    TSMixer = tsmixer_model_module.TSMixer
    build_tsmixer_features = tsmixer_features_module.build_features
    TSMIXER_FEATURE_COLS = tsmixer_features_module.FEATURE_COLS
    TSMIXER_TARGET_IDX = TSMIXER_FEATURE_COLS.index("log_return")
except Exception as exc:
    LSTMForecaster = None
    build_lstm_features = None
    TSMixer = None
    build_tsmixer_features = None
    TSMIXER_FEATURE_COLS = []
    TSMIXER_TARGET_IDX = 0
    LEGACY_IMPORT_ERROR = exc

from sklearn.preprocessing import StandardScaler


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


tcn_preprocessing_module = import_from_path(
    "demo_tcn_preprocessing",
    os.path.join(SRC_PREPROCESSING_DIR, "tcn_baseline_preprocessing.py"),
)
calculate_bollinger_bands = tcn_preprocessing_module.calculate_bollinger_bands
calculate_rsi = tcn_preprocessing_module.calculate_rsi
calculate_macd = tcn_preprocessing_module.calculate_macd

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
TFT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'tft_config.yaml')
DEMO_MODEL_PATHS = {
    "TFT": os.path.join(PROJECT_ROOT, "experiments", "tft"),
    "TFT-FinBERT": os.path.join(PROJECT_ROOT, "experiments", "tft_finbert"),
}
LSTM_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "experiments", "lstm_results", "model.pt")
TSMIXER_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "experiments", "tsmixer_results", "global.pt")
MODEL_CHOICES = [
    "TFT",
    "TFT-FinBERT",
    "LSTM",
    "TSMixer",
]
TFT_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
]


def resolve_demo_experiment_path(model_type, config):
    if model_type in ("TFT", "Standalone TFT (Numerical)"):
        return DEMO_MODEL_PATHS["TFT"]
    if model_type in ("TFT-FinBERT", "TFT-FinBERT (Multimodal)"):
        return DEMO_MODEL_PATHS["TFT-FinBERT"]
    return config.get("experiment_path")

SELECT_PLACEHOLDER = "Select"

BG_DEEP = "#090B10"
BG_PANEL = "#121722"
GRID = "#263042"
ZERO = "#3B465A"
ACCENT_ORANGE = "#FF8A3D"
ACCENT_BLUE = "#62A8FF"
ACCENT_GREEN = "#30D47F"
ACCENT_RED = "#FF5C5C"
TEXT_BODY = "#E8EDF7"
TEXT_HEAD = "#FFFFFF"
TEXT_MUTED = "#9BA8BC"
MONO_FONT = "'IBM Plex Mono', 'Courier New', Consolas, monospace"

st.set_page_config(
    page_title="Stock Backtesting Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Inter:wght@400;600&display=swap');

    .stApp {{
        background-color: {BG_DEEP};
        color: {TEXT_BODY};
    }}
    .block-container {{
        padding-top: 3.5rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_HEAD} !important;
        font-family: {MONO_FONT};
        letter-spacing: 0.04em;
        font-weight: 600;
    }}
    p, span, div, label {{
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
    }}
    .terminal-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid {ACCENT_ORANGE};
        border-bottom: 1px solid {ACCENT_ORANGE};
        padding: 6px 12px;
        margin-bottom: 12px;
        background-color: {BG_PANEL};
    }}
    .terminal-title {{
        color: {ACCENT_ORANGE};
        font-family: {MONO_FONT};
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }}
    .terminal-meta {{
        color: {TEXT_MUTED};
        font-family: {MONO_FONT};
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {BG_PANEL};
        border-right: 1px solid {GRID};
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {ACCENT_ORANGE} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid {GRID};
        padding-bottom: 4px;
    }}
    section[data-testid="stSidebar"] label {{
        color: {TEXT_MUTED} !important;
        font-family: {MONO_FONT} !important;
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {BG_DEEP};
        border: 1px solid {GRID};
        border-radius: 0;
        font-family: {MONO_FONT};
        color: {TEXT_BODY};
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] div {{
        color: {TEXT_BODY};
    }}
    div[data-testid="stMetric"] {{
        background-color: {BG_PANEL};
        border: 1px solid {GRID};
        border-left: 2px solid {ACCENT_ORANGE};
        padding: 8px 12px;
        border-radius: 0;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {TEXT_MUTED} !important;
        font-family: {MONO_FONT} !important;
        font-size: 0.66rem !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: {MONO_FONT} !important;
        color: {TEXT_HEAD} !important;
        font-size: 1.4rem !important;
        font-weight: 700;
    }}
    div[data-testid="stMetricDelta"] {{
        font-family: {MONO_FONT} !important;
        font-size: 0.78rem !important;
    }}
    .stButton > button {{
        background-color: {BG_PANEL};
        color: {ACCENT_ORANGE};
        border: 1px solid {ACCENT_ORANGE};
        border-radius: 0;
        font-family: {MONO_FONT};
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 0.45rem 1rem;
        width: 100%;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: {ACCENT_ORANGE};
        color: {BG_DEEP};
        box-shadow: 0 0 0 1px {ACCENT_ORANGE};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {BG_PANEL};
        border-bottom: 1px solid {GRID};
        gap: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {TEXT_MUTED};
        font-family: {MONO_FONT};
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 8px 16px;
        border-radius: 0;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {ACCENT_ORANGE};
        border-bottom: 2px solid {ACCENT_ORANGE};
        background-color: {BG_DEEP};
    }}
    .stDataFrame, .stDataFrame [data-testid="stTable"] {{
        background-color: {BG_PANEL};
        font-family: {MONO_FONT} !important;
    }}
    div[data-testid="stTextInput"] input,
    textarea,
    input {{
        background-color: {BG_DEEP} !important;
        color: {TEXT_BODY} !important;
        border-color: {GRID} !important;
    }}
    div[data-testid="stMarkdownContainer"] p {{
        font-size: 1rem;
        line-height: 1.55;
    }}
    div[data-testid="stAlert"] {{
        background-color: {BG_PANEL};
        border: 1px solid {GRID};
        border-left: 2px solid {ACCENT_BLUE};
        border-radius: 0;
        font-family: {MONO_FONT};
        font-size: 0.8rem;
        color: {TEXT_BODY};
    }}
    div[data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
    div[data-baseweb="notification"][kind="warning"] {{
        border-left-color: {ACCENT_ORANGE};
    }}
    div[data-baseweb="notification"][kind="error"] {{
        border-left-color: {ACCENT_RED};
    }}
    div[data-baseweb="notification"][kind="success"] {{
        border-left-color: {ACCENT_GREEN};
    }}
    .stSpinner > div {{
        border-top-color: {ACCENT_ORANGE} !important;
    }}
</style>
""", unsafe_allow_html=True)


def apply_terminal_layout(fig, height=380, show_legend=True):
    fig.update_layout(
        template="plotly",
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_DEEP,
        font=dict(family=MONO_FONT, size=11, color=TEXT_BODY),
        margin=dict(l=50, r=20, t=45, b=40),
        height=height,
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor=BG_PANEL,
            bordercolor=GRID,
            borderwidth=1,
            font=dict(size=9, color=TEXT_BODY, family=MONO_FONT),
        ),
        hoverlabel=dict(
            bgcolor=BG_PANEL,
            bordercolor=ACCENT_ORANGE,
            font=dict(family=MONO_FONT, color=TEXT_BODY, size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor=ZERO, linecolor=GRID,
        tickfont=dict(color=TEXT_MUTED, size=10, family=MONO_FONT),
        title=dict(font=dict(color=TEXT_MUTED, size=10, family=MONO_FONT)),
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=ZERO, linecolor=GRID,
        tickfont=dict(color=TEXT_MUTED, size=10, family=MONO_FONT),
        title=dict(font=dict(color=TEXT_MUTED, size=10, family=MONO_FONT)),
    )
    return fig


def render_terminal_header(ticker, model_type):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f"""
        <div class="terminal-bar">
            <div class="terminal-title">{ticker} &middot; {model_type}</div>
            <div class="terminal-meta">SESSION {timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, days=150):
    ticker = ticker.upper().strip()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)

    attempts = (
        lambda: yf.download(
            ticker,
            period=f"{max(days, 30)}d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        ),
        lambda: yf.Ticker(ticker).history(
            start=start_date,
            end=end_date + datetime.timedelta(days=1),
            interval="1d",
            auto_adjust=False,
        ),
    )

    for attempt in attempts:
        try:
            df = normalize_stock_frame(attempt())
            if not df.empty:
                return df
        except Exception:
            continue

    return fetch_stock_data_from_yahoo_chart(ticker, days)


def normalize_stock_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")


def fetch_stock_data_from_yahoo_chart(ticker, days=150):
    end_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    start_ts = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).timestamp())
    query = urlencode({
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "includePrePost": "false",
        "events": "history",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return pd.DataFrame()

    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame()

    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = ((chart.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps or not quote:
        return pd.DataFrame()

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "volume": quote.get("volume"),
    })
    return normalize_stock_frame(df)


@st.cache_data(ttl=3600)
def fetch_news_data(ticker, max_articles=4):
    try:
        news_items = yf.Ticker(ticker).news
    except Exception:
        news_items = []

    articles = []

    for item in news_items:
        title = item.get('title', '')
        if title:
            articles.append(title)
        if len(articles) >= max_articles:
            break

    while len(articles) < max_articles:
        articles.append("")

    return articles


def preprocess_data(df):
    return preprocess_tft_input_window(df)


def preprocess_tft_input_window(df):
    df = df.copy()
    close_prices = df['close']

    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['rsi'] = calculate_rsi(close_prices)
    macd_line, signal_line, histogram = calculate_macd(close_prices)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram

    df = df.ffill().bfill().fillna(0)
    return df


def fit_tft_input_scaler(model_type, model_input_data, processed_df, features, experiment_path):
    if "TFT-FinBERT" in model_type:
        scaler = StandardScaler()
        scaler.fit(model_input_data[features].values)
        return scaler

    scaler_path = os.path.join(experiment_path, 'preprocessed_data.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            pp = pickle.load(f)
        scaler = pp.get('scaler')
        if scaler is not None:
            return scaler

    scaler = StandardScaler()
    scaler.fit(processed_df[features].values)
    return scaler


def get_available_models():
    return MODEL_CHOICES


def should_wait_for_run_click(model_choice, ticker, run_clicked):
    return (
        model_choice != SELECT_PLACEHOLDER
        and ticker != SELECT_PLACEHOLDER
        and not run_clicked
    )


def get_missing_dependency_messages():
    messages = []
    if LEGACY_IMPORT_ERROR is not None:
        messages.append(f"Legacy model adapters unavailable: {LEGACY_IMPORT_ERROR}")
    if not os.path.exists(LSTM_CHECKPOINT_PATH):
        messages.append(f"Missing LSTM checkpoint: {LSTM_CHECKPOINT_PATH}")
    if not os.path.exists(TSMIXER_CHECKPOINT_PATH):
        messages.append(f"Missing TSMixer checkpoint: {TSMIXER_CHECKPOINT_PATH}")
    return messages


def to_legacy_ohlcv(df):
    legacy = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }).copy()
    legacy["Date"] = pd.to_datetime(legacy["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    legacy = legacy.dropna(subset=["Date"]).sort_values("Date")
    return legacy[["Date", "Open", "High", "Low", "Close", "Volume"]]


def close_by_date(legacy_df):
    close = legacy_df.set_index("Date")["Close"].astype(float).sort_index()
    close.index = pd.DatetimeIndex(close.index).normalize()
    return close


def reconstruct_prices(last_close, log_returns):
    return last_close * np.exp(np.cumsum(np.asarray(log_returns, dtype=float)))


def fmt_date(d):
    return pd.Timestamp(d).strftime("%b %d")


def get_valid_backtest_start_dates(df, seq_len, horizon):
    if len(df) < seq_len + horizon:
        return []
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    return list(dates.iloc[seq_len:len(df) - horizon + 1])


def get_latest_backtest_start_date(valid_backtest_dates):
    if not valid_backtest_dates:
        raise ValueError("No valid backtest start dates are available.")
    return valid_backtest_dates[-1]


def find_date_position(dates, target_date):
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    target = pd.Timestamp(target_date).normalize()
    matches = np.flatnonzero(normalized == target)
    if len(matches) == 0:
        return None
    return int(matches[0])


def get_model_backtest_start_dates(model_type, df, processed_df, seq_len, horizon):
    if "LSTM" in model_type:
        legacy_df = to_legacy_ohlcv(df)
        features_df = build_lstm_features(legacy_df.set_index("Date"))
        lookback = seq_len
        if os.path.exists(LSTM_CHECKPOINT_PATH):
            lookback = int(torch.load(LSTM_CHECKPOINT_PATH, map_location="cpu", weights_only=False).get("lookback", seq_len))
        return get_valid_backtest_start_dates(
            pd.DataFrame({"date": pd.to_datetime(features_df.index)}),
            lookback,
            horizon,
        )
    if "TSMixer" in model_type:
        legacy_df = to_legacy_ohlcv(df)
        features_df = build_tsmixer_features(legacy_df)
        return get_valid_backtest_start_dates(
            pd.DataFrame({"date": pd.to_datetime(features_df.index)}),
            60,
            horizon,
        )
    return get_valid_backtest_start_dates(processed_df, seq_len, horizon)


def prepare_backtest_window(df, seq_len, horizon, backtest_start_date):
    frame = df.copy().reset_index(drop=True)
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    start_date = pd.Timestamp(backtest_start_date).normalize()
    matches = np.flatnonzero(dates.eq(start_date).to_numpy())
    if len(matches) == 0:
        raise ValueError(f"Backtest start date {start_date:%Y-%m-%d} is not available in the data.")

    start_idx = int(matches[0])
    context_start = start_idx - seq_len
    actual_end = start_idx + horizon
    if context_start < 0 or actual_end > len(frame):
        raise ValueError("Selected backtest window does not have enough context or ground truth.")
    return frame.iloc[context_start:start_idx].copy(), frame.iloc[start_idx:actual_end].copy()


def compute_backtest_metrics(context_last_price, actual_prices, predicted_prices):
    actual = np.asarray(actual_prices, dtype=float)
    predicted = np.asarray(predicted_prices, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must have the same length.")

    errors = predicted - actual
    actual_path = np.concatenate([[float(context_last_price)], actual])
    predicted_path = np.concatenate([[float(context_last_price)], predicted])
    actual_direction = np.sign(np.diff(actual_path))
    predicted_direction = np.sign(np.diff(predicted_path))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "directional_accuracy": float(np.mean(actual_direction == predicted_direction) * 100),
    }


def build_backtest_figure(history_dates, history_prices, backtest_dates, actual_prices, pred_prices):
    pivot_date = history_dates[-1]
    pivot_price = history_prices[-1]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_dates, y=history_prices,
        mode="lines+markers",
        name="Historical Context",
        line=dict(color=ACCENT_BLUE, width=1.5),
        marker=dict(size=3, color=ACCENT_BLUE),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    actual_x = [pivot_date] + list(backtest_dates)
    actual_y = [pivot_price] + list(actual_prices)
    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_y,
        mode="lines+markers",
        name="Actual",
        line=dict(color=ACCENT_GREEN, width=1.7),
        marker=dict(size=4, color=ACCENT_GREEN),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    pred_x = [pivot_date] + list(backtest_dates)
    pred_y = [pivot_price] + list(pred_prices)
    fig.add_trace(go.Scatter(
        x=pred_x, y=pred_y,
        mode="lines+markers",
        name="Predicted",
        line=dict(color=ACCENT_ORANGE, width=1.5, dash="dot"),
        marker=dict(size=4, color=ACCENT_ORANGE),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    fig.update_yaxes(title="PRICE (USD)")
    fig.update_xaxes(title="DATE")
    fig.add_shape(
        type="line",
        x0=pivot_date,
        x1=pivot_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=TEXT_MUTED, width=1, dash="dash"),
    )
    fig.add_annotation(
        x=pivot_date,
        y=1,
        xref="x",
        yref="paper",
        text="Backtest Start",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(color=TEXT_MUTED, size=8, family=MONO_FONT),
    )
    return apply_terminal_layout(fig)


def show_backtest_result(ticker, model_type, history_dates, history_prices, backtest_dates, actual_prices, pred_prices):
    start_price = float(history_prices[-1])
    actual_close = float(actual_prices[-1])
    forecast_close = float(pred_prices[-1])
    actual_return = (actual_close - start_price) / start_price * 100
    predicted_return = (forecast_close - start_price) / start_price * 100
    start_label = fmt_date(history_dates[-1])

    render_terminal_header(ticker, f"{model_type} Backtest")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(f"Price {start_label}", f"${start_price:,.2f}")
    with k2:
        st.metric("Actual Close", f"${actual_close:,.2f}", delta=f"{actual_return:+.2f}%")
    with k3:
        st.metric("Forecast Close", f"${forecast_close:,.2f}", delta=f"{predicted_return:+.2f}%")

    chart_tab, data_tab = st.tabs(["Backtest", "Data"])

    with chart_tab:
        fig = build_backtest_figure(history_dates, history_prices, backtest_dates, actual_prices, pred_prices)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with data_tab:
        backtest_df = pd.DataFrame({
            "Date": [d.strftime('%Y-%m-%d') for d in backtest_dates],
            "Actual": [f"${p:,.2f}" for p in actual_prices],
            "Predicted": [f"${p:,.2f}" for p in pred_prices],
            "Abs Error": [f"${abs(a - p):,.2f}" for a, p in zip(actual_prices, pred_prices)],
        })
        st.dataframe(backtest_df, use_container_width=True, hide_index=True, height=240)


@st.cache_resource
def load_model(config, model_type="TFT"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_path = None

    if model_type in ("TFT", "Standalone TFT (Numerical)"):
        if TFTModel is None:
            raise RuntimeError(f"TFT model definition is unavailable: {TFT_IMPORT_ERROR}")
        model_config = config.get('model', {})
        model = TFTModel(model_config).to(device)
        experiment_path = resolve_demo_experiment_path(model_type, config)
    elif model_type in ("TFT-FinBERT", "TFT-FinBERT (Multimodal)"):
        if TftMultiModalStockPredictor is None:
            raise RuntimeError(f"Multimodal TFT definition is unavailable: {MULTIMODAL_IMPORT_ERROR}")
        model = TftMultiModalStockPredictor(config, num_tickers=234, num_sectors=13).to(device)
        experiment_path = resolve_demo_experiment_path(model_type, config)
    elif "LSTM" in model_type:
        if LSTMForecaster is None:
            raise RuntimeError(f"LSTM runtime is unavailable: {LEGACY_IMPORT_ERROR}")
        ckpt = torch.load(LSTM_CHECKPOINT_PATH, map_location=device, weights_only=False)
        model_cfg = ckpt.get("model_cfg", {})
        model = LSTMForecaster(
            n_features=int(ckpt["n_features"]),
            horizon=int(ckpt["horizon"]),
            hidden1=int(model_cfg.get("hidden1", 128)),
            hidden2=int(model_cfg.get("hidden2", 64)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            output_bound=model_cfg.get("output_bound"),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, True, device, {
            "checkpoint": ckpt,
            "experiment_path": os.path.dirname(LSTM_CHECKPOINT_PATH),
        }
    elif "TSMixer" in model_type:
        if TSMixer is None:
            raise RuntimeError(f"TSMixer runtime is unavailable: {LEGACY_IMPORT_ERROR}")
        ckpt = torch.load(TSMIXER_CHECKPOINT_PATH, map_location=device, weights_only=False)
        ticker_to_id = ckpt.get("ticker_to_id", {})
        model = TSMixer(
            lookback=60,
            n_features=len(ckpt.get("feature_cols", TSMIXER_FEATURE_COLS)),
            horizon=5,
            target_idx=TSMIXER_TARGET_IDX,
            n_blocks=4,
            ff_dim=128,
            dropout=0.1,
            num_tickers=len(ticker_to_id),
            ticker_embed_dim=int(ckpt.get("ticker_embed_dim", 8)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, True, device, {
            "checkpoint": ckpt,
            "experiment_path": os.path.dirname(TSMIXER_CHECKPOINT_PATH),
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    best_model_path = os.path.join(experiment_path, 'checkpoints', 'best_model.pth')
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(experiment_path, 'best_model.pth')

    model_loaded = False
    if os.path.exists(best_model_path):
        try:
            state_dict = torch.load(best_model_path, map_location=device)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model_loaded = True
        except Exception as e:
            st.warning(f"Failed to load weights for {model_type}: {e}")

    model.eval()
    return model, model_loaded, device, experiment_path


def run_lstm_backtest(model, model_context, device, df, ticker, backtest_start_date):
    ckpt = model_context["checkpoint"]
    legacy_df = to_legacy_ohlcv(df)
    features_df = build_lstm_features(legacy_df.set_index("Date"))
    feature_names = ckpt["feature_names"]
    missing = [name for name in feature_names if name not in features_df.columns]
    if missing:
        raise ValueError(f"LSTM features missing columns: {missing}")

    lookback = int(ckpt["lookback"])
    horizon = int(ckpt["horizon"])
    dates = pd.DatetimeIndex(pd.to_datetime(features_df.index).normalize())
    context_end = find_date_position(dates, backtest_start_date)
    if context_end is None:
        start_date = pd.Timestamp(backtest_start_date).normalize()
        raise ValueError(f"Backtest start date {start_date:%Y-%m-%d} is not available after LSTM feature engineering.")

    context_start = context_end - lookback
    actual_end = context_end + horizon
    if context_start < 0 or actual_end > len(features_df):
        raise ValueError("Selected backtest window does not have enough LSTM context or ground truth.")

    window = features_df[feature_names].to_numpy()[context_start:context_end]

    x = ckpt["feat_scaler"].transform(window).astype(np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    pred_returns = ckpt["target_scaler"].inverse_transform(pred_scaled.reshape(1, -1)).ravel()

    close = close_by_date(legacy_df)
    context_dates = list(dates[context_start:context_end])
    actual_dates = list(dates[context_end:actual_end])
    history_dates = context_dates[-30:]
    history_prices = close.reindex(pd.DatetimeIndex(history_dates)).ffill().to_numpy(dtype=float)
    actual_prices = close.reindex(pd.DatetimeIndex(actual_dates)).ffill().to_numpy(dtype=float)
    pred_prices = reconstruct_prices(float(history_prices[-1]), pred_returns)
    if not np.isfinite(pred_prices).all() or not np.isfinite(actual_prices).all():
        raise ValueError("LSTM produced non-finite predicted prices.")
    show_backtest_result(ticker, "LSTM", history_dates, history_prices, actual_dates, actual_prices, pred_prices)


def run_tsmixer_backtest(model, model_context, device, df, ticker, backtest_start_date):
    ckpt = model_context["checkpoint"]
    legacy_df = to_legacy_ohlcv(df)
    features_df = build_tsmixer_features(legacy_df)
    feature_cols = ckpt.get("feature_cols", TSMIXER_FEATURE_COLS)
    missing = [name for name in feature_cols if name not in features_df.columns]
    if missing:
        raise ValueError(f"TSMixer features missing columns: {missing}")

    lookback = 60
    horizon = 5
    raw_x = features_df[feature_cols].to_numpy(dtype=np.float32)

    ticker_key = ticker.lower()
    target_scalers = ckpt.get("target_scalers", {})
    scaler_stats = target_scalers.get(ticker_key, {"center": 0.0, "scale": 1.0})
    center = float(scaler_stats.get("center", 0.0))
    scale = float(scaler_stats.get("scale", 1.0)) or 1.0

    x = raw_x.copy()
    x[:, TSMIXER_TARGET_IDX] = ((x[:, TSMIXER_TARGET_IDX] - center) / scale).astype(np.float32)

    dates = pd.DatetimeIndex(pd.to_datetime(features_df.index).normalize())
    context_end = find_date_position(dates, backtest_start_date)
    if context_end is None:
        start_date = pd.Timestamp(backtest_start_date).normalize()
        raise ValueError(f"Backtest start date {start_date:%Y-%m-%d} is not available after TSMixer feature engineering.")

    context_start = context_end - lookback
    actual_end = context_end + horizon
    if context_start < 0 or actual_end > len(x):
        raise ValueError("Selected backtest window does not have enough TSMixer context or ground truth.")

    window = x[context_start:context_end]

    ticker_to_id = ckpt.get("ticker_to_id", {})
    ticker_id = torch.tensor([int(ticker_to_id.get(ticker_key, 0))], device=device, dtype=torch.long)
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(window).unsqueeze(0).to(device), ticker_id=ticker_id).cpu().numpy()[0]
    pred_returns = pred_scaled * scale + center

    close = close_by_date(legacy_df)
    history_dates = list(dates[max(context_start, context_end - 30):context_end])
    actual_dates = list(dates[context_end:actual_end])
    history_prices = close.reindex(pd.DatetimeIndex(history_dates)).ffill().to_numpy(dtype=float)
    actual_prices = close.reindex(pd.DatetimeIndex(actual_dates)).ffill().to_numpy(dtype=float)
    pred_prices = reconstruct_prices(float(history_prices[-1]), pred_returns)
    if not np.isfinite(pred_prices).all() or not np.isfinite(actual_prices).all():
        raise ValueError("TSMixer produced non-finite predicted prices.")
    show_backtest_result(ticker, "TSMixer", history_dates, history_prices, actual_dates, actual_prices, pred_prices)


def main():
    st.markdown(
        f"""
        <div class="terminal-bar">
            <div class="terminal-title">STOCK BACKTESTING TERMINAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Configuration")

    available_models = get_available_models()
    missing_messages = get_missing_dependency_messages()
    if missing_messages:
        st.sidebar.warning("Some demo components are unavailable in this checkout.")
        with st.sidebar.expander("Startup diagnostics"):
            for message in missing_messages:
                st.write(f"- {message}")

    popular_stocks = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "BRK-B", "JPM", "V",
        "WMT", "JNJ", "PG", "MA", "HD"
    ]
    model_choice = st.sidebar.selectbox("MODEL", [SELECT_PLACEHOLDER] + available_models)
    ticker = st.sidebar.selectbox("TICKER", options=[SELECT_PLACEHOLDER] + popular_stocks)
    run_clicked = st.sidebar.button("Run Backtest")

    if model_choice == SELECT_PLACEHOLDER or ticker == SELECT_PLACEHOLDER:
        st.markdown(
            f"<div class='terminal-meta' style='padding:24px 4px;'>READY · Select a model and ticker in the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return

    if "TFT-FinBERT" in model_choice and not TRANSFORMERS_AVAILABLE:
        st.sidebar.error("Transformers library not available. Start Streamlit with:\n`python3 -m streamlit run demo/app.py --server.fileWatcherType none`")
        return

    if should_wait_for_run_click(model_choice, ticker, run_clicked):
        st.markdown(
            f"<div class='terminal-meta' style='padding:50px 24px;'>BACKTEST READY &middot; Press <span style='color:{ACCENT_ORANGE}'>RUN BACKTEST</span> to fetch data and evaluate the most recent historical window.</div>",
            unsafe_allow_html=True,
        )
        return

    backtest_horizon = 5
    try:
        yaml_config = read_yaml(CONFIG_PATH)
        tft_config = read_yaml(TFT_CONFIG_PATH) if os.path.exists(TFT_CONFIG_PATH) else {}
        config = {**tft_config, **yaml_config}
        config.setdefault('model', {})
        config['model']['output_size'] = backtest_horizon
        config['num_articles'] = 4
    except Exception as e:
        st.error(f"Error loading configs: {e}")
        return

    with st.spinner(f"Fetching data & indicators for {ticker}..."):
        df = fetch_stock_data(ticker, days=150)

    if df.empty:
        st.error(f"No data found for ticker {ticker}.")
        return

    processed_df = preprocess_data(df)
    seq_len = int(config.get('HISTORY_WINDOW_SIZE', 60))
    valid_backtest_dates = get_model_backtest_start_dates(model_choice, df, processed_df, seq_len, backtest_horizon)
    if not valid_backtest_dates:
        st.error("Not enough data to create a backtest window with full ground truth.")
        return

    backtest_start_date = get_latest_backtest_start_date(valid_backtest_dates)

    is_multimodal = "TFT-FinBERT" in model_choice
    with st.spinner(f"Loading {model_choice}..."):
        try:
            model, model_loaded, device, experiment_path = load_model(config, model_choice)
        except RuntimeError as e:
            st.info(str(e))
            return
        model_type = model_choice.split(" ")[0]

    if not model_loaded:
        st.sidebar.warning(f"No pre-trained weights for {model_type}. Using untrained baseline.")
    else:
        st.sidebar.success("weights loaded.")

    if "LSTM" in model_choice:
        try:
            run_lstm_backtest(model, experiment_path, device, df, ticker, backtest_start_date)
        except Exception as e:
            st.error(f"LSTM backtest failed: {e}")
        return

    if "TSMixer" in model_choice:
        try:
            run_tsmixer_backtest(model, experiment_path, device, df, ticker, backtest_start_date)
        except Exception as e:
            st.error(f"TSMixer backtest failed: {e}")
        return

    tokenized_news = None
    attention_mask = None
    news_articles = []
    if is_multimodal:
        with st.spinner(f"Fetching news context for {ticker}..."):
            news_articles = fetch_news_data(ticker, max_articles=4)

            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            encoded = tokenizer(
                news_articles,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            tokenized_news = encoded['input_ids'].unsqueeze(0).to(device)
            attention_mask = encoded['attention_mask'].unsqueeze(0).to(device)

    try:
        model_input_data, actual_data = prepare_backtest_window(
            df, seq_len, backtest_horizon, backtest_start_date
        )
    except ValueError as e:
        st.error(str(e))
        return

    model_input_data = preprocess_tft_input_window(model_input_data)
    features = TFT_FEATURES
    X_raw = model_input_data[features].values

    scaler = fit_tft_input_scaler(model_type, model_input_data, processed_df, features, experiment_path)
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with st.spinner("Generating Backtest..."):
        with torch.no_grad():
            if is_multimodal:
                ticker_id = torch.tensor([0], dtype=torch.long).to(device)
                sector_id = torch.tensor([0], dtype=torch.long).to(device)

                preds_scaled = model(
                    tokenized_news_=tokenized_news,
                    attention_mask_news_=attention_mask,
                    time_series_features_=X_tensor,
                    ticker_id_=ticker_id,
                    sector_=sector_id
                ).cpu().numpy()[0]
            else:
                preds_scaled = model(X_tensor).cpu().numpy()[0]

        close_idx = features.index('close')
        close_mean = scaler.mean_[close_idx]
        close_std = scaler.scale_[close_idx]
        preds_raw = (preds_scaled * close_std) + close_mean

    if not np.isfinite(preds_raw).all():
        st.error(f"{model_type} produced non-finite predicted prices.")
        return

    plot_days = 30
    history_dates = model_input_data['date'].tolist()[-plot_days:]
    history_prices = model_input_data['close'].values[-plot_days:]
    actual_dates = actual_data['date'].tolist()
    actual_prices = actual_data['close'].values
    show_backtest_result(ticker, model_type, history_dates, history_prices, actual_dates, actual_prices, preds_raw)

    if is_multimodal and news_articles:
        with st.expander("News Context"):
            for i, article in enumerate(news_articles):
                if article:
                    st.markdown(
                        f"<div style='border-left:2px solid {ACCENT_BLUE};background:{BG_PANEL};"
                        f"padding:8px 12px;margin-bottom:6px;font-family:{MONO_FONT};font-size:0.82rem;color:{TEXT_BODY};'>"
                        f"<span style='color:{TEXT_MUTED};font-size:0.7rem;'>ARTICLE {i+1:02d}</span><br>{article}</div>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
