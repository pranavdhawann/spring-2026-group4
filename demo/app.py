import os
import sys
import pickle
import datetime
import importlib.util
import json
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SRC_MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "models")
SRC_PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, "src", "preProcessing")
LSTM_LEGACY_DIR = os.path.join(PROJECT_ROOT, "cookbooks", "legacy", "lstm")
TSMIXER_LEGACY_DIR = os.path.join(PROJECT_ROOT, "cookbooks", "legacy", "tsmixer")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Add project root AND src/models to path to bypass __init__.py
for path in (PROJECT_ROOT, SRC_MODELS_DIR):
    if path not in sys.path:
        sys.path.append(path)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

TFT_IMPORT_ERROR = None
MULTIMODAL_IMPORT_ERROR = None
LEGACY_IMPORT_ERROR = None

try:
    from tft_model import TFTModel
except ImportError as exc:
    TFTModel = None
    TFT_IMPORT_ERROR = exc

try:
    from TftMultiModalBaseline import MultiModalStockPredictor as TftMultiModalStockPredictor
except ImportError as exc:
    TftMultiModalStockPredictor = None
    MULTIMODAL_IMPORT_ERROR = exc


def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    "Standalone TFT (Numerical)": os.path.join(BASE_DIR, 'models', 'tft'),
    "TFT-FinBERT (Multimodal)": os.path.join(BASE_DIR, 'models', 'tft_finbert'),
}
LSTM_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "experiments", "lstm_results", "model.pt")
LSTM_METRICS_PATH = os.path.join(PROJECT_ROOT, "experiments", "lstm_results", "test_metrics.csv")
TSMIXER_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "experiments", "tsmixer_results", "global.pt")
TSMIXER_METRICS_PATH = os.path.join(PROJECT_ROOT, "experiments", "tsmixer_results", "results.json")
MODEL_CHOICES = [
    "Standalone TFT (Numerical)",
    "TFT-FinBERT (Multimodal)",
    "LSTM (Legacy)",
    "TSMixer (Legacy)",
]

BG_DEEP = "#F0F2F5"
BG_PANEL = "#FFFFFF"
GRID = "#E2E6EA"
ZERO = "#C8CDD4"
ACCENT_ORANGE = "#E85D00"
ACCENT_BLUE = "#0066CC"
ACCENT_GREEN = "#007A3D"
ACCENT_RED = "#CC2200"
TEXT_BODY = "#1A1A1A"
TEXT_HEAD = "#000000"
TEXT_MUTED = "#5A6472"
MONO_FONT = "'IBM Plex Mono', 'Courier New', Consolas, monospace"

st.set_page_config(
    page_title="Stock Forecasting Terminal",
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
        padding-top: 1rem;
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
    hr {{
        border-color: {GRID};
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
        margin=dict(l=50, r=20, t=20, b=40),
        height=height,
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=GRID,
            borderwidth=1,
            font=dict(size=10, color=TEXT_BODY, family=MONO_FONT),
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
    """Fetch stock data from yfinance, with a Yahoo chart fallback."""
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
    """Fetch recent news articles using yfinance."""
    stock = yf.Ticker(ticker)
    news_items = stock.news
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
    """Calculate indicators and format data."""
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


def get_available_models():
    # Keep the TFT entries visible for the demo even when their model-definition
    # files are not present in this checkout.
    return MODEL_CHOICES


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


def build_future_dates(last_date, horizon):
    last_date = pd.Timestamp(last_date).normalize()
    return list(pd.bdate_range(last_date + pd.offsets.BDay(1), periods=horizon))


def build_price_rmse_band(last_close, predicted_prices, rmse_returns):
    predicted = np.asarray(predicted_prices, dtype=float)
    rmse = np.asarray(rmse_returns, dtype=float)
    if len(rmse) == 0:
        return None, None
    if len(rmse) < len(predicted):
        rmse = np.pad(rmse, (0, len(predicted) - len(rmse)), mode="edge")
    rmse = rmse[:len(predicted)]
    cumulative_rmse = np.sqrt(np.cumsum(np.square(rmse)))
    upper = predicted * np.exp(cumulative_rmse)
    lower = predicted * np.exp(-cumulative_rmse)
    return lower, upper


def build_rmse_fill_series(pivot_date, pivot_price, future_dates, pred_prices, rmse_returns):
    lower, upper = build_price_rmse_band(pivot_price, pred_prices, rmse_returns)
    if lower is None:
        return None, None, None
    dates = [pd.Timestamp(pivot_date)] + list(future_dates)
    lower = np.concatenate([[float(pivot_price)], lower])
    upper = np.concatenate([[float(pivot_price)], upper])
    return dates, lower, upper


def load_lstm_rmse_per_step(horizon):
    if not os.path.exists(LSTM_METRICS_PATH):
        return None
    metrics = pd.read_csv(LSTM_METRICS_PATH)
    column = "RMSE_original" if "RMSE_original" in metrics.columns else "RMSE"
    if column not in metrics.columns:
        return None
    return metrics[column].astype(float).head(horizon).tolist()


def load_tsmixer_rmse_per_step(horizon):
    if not os.path.exists(TSMIXER_METRICS_PATH):
        return None
    with open(TSMIXER_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rmse = metrics.get("test_agg", {}).get("RMSE")
    if rmse is None:
        return None
    return [float(rmse)] * horizon


def build_forward_forecast_figure(history_dates, history_prices, future_dates, pred_prices, rmse_returns=None):
    pivot_date = history_dates[-1]
    pivot_price = history_prices[-1]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_dates, y=history_prices,
        mode="lines+markers",
        name="HISTORICAL",
        line=dict(color=ACCENT_BLUE, width=1.5),
        marker=dict(size=3, color=ACCENT_BLUE),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    forecast_x = [pivot_date] + list(future_dates)
    forecast_y = [pivot_price] + list(pred_prices)
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y,
        mode="lines+markers",
        name="FORECAST",
        line=dict(color=ACCENT_ORANGE, width=1.5, dash="dot"),
        marker=dict(size=4, color=ACCENT_ORANGE),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    if rmse_returns:
        band_dates, lower, upper = build_rmse_fill_series(pivot_date, pivot_price, future_dates, pred_prices, rmse_returns)
        if lower is not None:
            fig.add_trace(go.Scatter(
                x=list(band_dates) + list(band_dates)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself",
                fillcolor="rgba(255,102,0,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="P10–P90",
                hoverinfo="skip",
            ))

    fig.update_yaxes(title="PRICE (USD)")
    fig.update_xaxes(title="DATE")
    return apply_terminal_layout(fig)


def build_backtest_figure(history_dates, history_prices, future_dates, actual_prices, pred_prices):
    pivot_date = history_dates[-1]
    pivot_price = history_prices[-1]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_dates, y=history_prices,
        mode="lines+markers",
        name="HISTORICAL",
        line=dict(color=ACCENT_BLUE, width=1.5),
        marker=dict(size=3, color=ACCENT_BLUE),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    actual_x = [pivot_date] + list(future_dates)
    actual_y = [pivot_price] + list(actual_prices)
    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_y,
        mode="lines+markers",
        name="ACTUAL",
        line=dict(color=ACCENT_GREEN, width=1.5),
        marker=dict(size=4, color=ACCENT_GREEN),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    pred_x = [pivot_date] + list(future_dates)
    pred_y = [pivot_price] + list(pred_prices)
    fig.add_trace(go.Scatter(
        x=pred_x, y=pred_y,
        mode="lines+markers",
        name="FORECAST",
        line=dict(color=ACCENT_ORANGE, width=1.5, dash="dot"),
        marker=dict(size=4, color=ACCENT_ORANGE),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    fig.update_yaxes(title="PRICE (USD)")
    fig.update_xaxes(title="DATE")
    return apply_terminal_layout(fig)


def fmt_date(d):
    return pd.Timestamp(d).strftime("%b %d")


def render_move_metric(label, price_delta, pct_change):
    color = ACCENT_GREEN if pct_change >= 0 else ACCENT_RED
    sign = "+" if pct_change >= 0 else ""
    st.markdown(
        f"<div style='background:{BG_PANEL};border:1px solid {GRID};border-left:2px solid {ACCENT_ORANGE};"
        f"padding:8px 12px;'>"
        f"<p style='color:{TEXT_MUTED};font-family:{MONO_FONT};font-size:0.66rem;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin:0 0 6px 0;'>{label}</p>"
        f"<p style='color:{color};font-family:{MONO_FONT};font-size:1.1rem;font-weight:700;margin:0;'>"
        f"{sign}${abs(price_delta):,.2f} &nbsp;·&nbsp; {sign}{pct_change:.2f}%</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def show_forward_forecast_result(ticker, model_type, history_dates, history_prices, future_dates, pred_prices, rmse_returns=None):
    current_price = float(history_prices[-1])
    predicted_end = float(pred_prices[-1])
    projected_change = (predicted_end - current_price) / current_price * 100
    price_delta = predicted_end - current_price

    render_terminal_header(ticker, model_type)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(f"Last Close  {fmt_date(history_dates[-1])}", f"${current_price:,.2f}")
    with k2:
        st.metric(f"Forecasted  {fmt_date(future_dates[-1])}", f"${predicted_end:,.2f}")
    with k3:
        render_move_metric("Projected Move", price_delta, projected_change)
    with k4:
        st.metric("Horizon", f"{len(pred_prices)} Days")

    chart_tab, data_tab = st.tabs(["Forecast", "Data"])

    with chart_tab:
        fig = build_forward_forecast_figure(history_dates, history_prices, future_dates, pred_prices, rmse_returns)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with data_tab:
        log_returns = np.log(np.asarray(pred_prices) / np.concatenate([[current_price], pred_prices[:-1]]))
        forecast_df = pd.DataFrame({
            "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
            "Predicted Price": [f"${p:,.2f}" for p in pred_prices],
            "Log Return": [f"{r:+.5f}" for r in log_returns],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True, height=240)


@st.cache_resource
def load_model(config, model_type="Standalone TFT (Numerical)"):
    """Load the selected model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "Standalone TFT" in model_type:
        if TFTModel is None:
            raise RuntimeError(f"TFT model definition is unavailable: {TFT_IMPORT_ERROR}")
        model_config = config.get('model', {})
        model = TFTModel(model_config).to(device)
        experiment_path = config.get('experiment_path', DEMO_MODEL_PATHS[model_type])
    elif "TFT-FinBERT" in model_type:
        if TftMultiModalStockPredictor is None:
            raise RuntimeError(f"Multimodal TFT definition is unavailable: {MULTIMODAL_IMPORT_ERROR}")
        model = TftMultiModalStockPredictor(config, num_tickers=234, num_sectors=13).to(device)
        experiment_path = config.get('experiment_path_tft_multimodal', DEMO_MODEL_PATHS[model_type])
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

    best_model_path = os.path.join(experiment_path, 'checkpoints', 'best_model.pth')
    if not os.path.exists(best_model_path):
        # Fallback to a flat directory layout used by some legacy checkouts.
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


def run_lstm_backtest(model, model_context, device, df, ticker):
    ckpt = model_context["checkpoint"]
    legacy_df = to_legacy_ohlcv(df)
    features_df = build_lstm_features(legacy_df.set_index("Date"))
    feature_names = ckpt["feature_names"]
    missing = [name for name in feature_names if name not in features_df.columns]
    if missing:
        raise ValueError(f"LSTM features missing columns: {missing}")

    lookback = int(ckpt["lookback"])
    horizon = int(ckpt["horizon"])
    if len(features_df) < lookback:
        raise ValueError(f"Need at least {lookback} engineered rows, got {len(features_df)}.")

    context_end = len(features_df)
    context_start = context_end - lookback
    window = features_df[feature_names].to_numpy()[context_start:context_end]

    x = ckpt["feat_scaler"].transform(window).astype(np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    pred_returns = ckpt["target_scaler"].inverse_transform(pred_scaled.reshape(1, -1)).ravel()

    close = close_by_date(legacy_df)
    context_dates = list(pd.to_datetime(features_df.index[context_start:context_end]).normalize())
    history_dates = context_dates[-30:]
    history_prices = close.reindex(pd.DatetimeIndex(history_dates)).ffill().to_numpy(dtype=float)
    pred_prices = reconstruct_prices(float(history_prices[-1]), pred_returns)
    future_dates = build_future_dates(history_dates[-1], horizon)
    if not np.isfinite(pred_prices).all():
        raise ValueError("LSTM produced non-finite predicted prices.")
    rmse_returns = load_lstm_rmse_per_step(horizon)
    show_forward_forecast_result(ticker, "LSTM", history_dates, history_prices, future_dates, pred_prices, rmse_returns)


def run_tsmixer_backtest(model, model_context, device, df, ticker):
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
    if len(raw_x) < lookback:
        raise ValueError(f"Need at least {lookback} engineered rows, got {len(raw_x)}.")

    ticker_key = ticker.lower()
    target_scalers = ckpt.get("target_scalers", {})
    scaler_stats = target_scalers.get(ticker_key, {"center": 0.0, "scale": 1.0})
    center = float(scaler_stats.get("center", 0.0))
    scale = float(scaler_stats.get("scale", 1.0)) or 1.0

    x = raw_x.copy()
    x[:, TSMIXER_TARGET_IDX] = ((x[:, TSMIXER_TARGET_IDX] - center) / scale).astype(np.float32)

    end = len(x) - 1
    start = end - lookback + 1
    window = x[start:end + 1]

    ticker_to_id = ckpt.get("ticker_to_id", {})
    ticker_id = torch.tensor([int(ticker_to_id.get(ticker_key, 0))], device=device, dtype=torch.long)
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(window).unsqueeze(0).to(device), ticker_id=ticker_id).cpu().numpy()[0]
    pred_returns = pred_scaled * scale + center

    close = close_by_date(legacy_df)
    history_dates = list(pd.to_datetime(features_df.index[max(0, end - 29):end + 1]).normalize())
    history_prices = close.reindex(pd.DatetimeIndex(history_dates)).ffill().to_numpy(dtype=float)
    pred_prices = reconstruct_prices(float(history_prices[-1]), pred_returns)
    future_dates = build_future_dates(history_dates[-1], horizon)
    if not np.isfinite(pred_prices).all():
        raise ValueError("TSMixer produced non-finite predicted prices.")
    rmse_returns = load_tsmixer_rmse_per_step(horizon)
    show_forward_forecast_result(ticker, "TSMixer", history_dates, history_prices, future_dates, pred_prices, rmse_returns)


def main():
    st.markdown(
        f"""
        <div class="terminal-bar">
            <div class="terminal-title">STOCK FORECASTING TERMINAL</div>
            <div class="terminal-meta">MULTI-MODEL · LIVE FEED · SENTIMENT</div>
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

    model_choice = st.sidebar.selectbox("Architecture", available_models)

    if "Multimodal" in model_choice and not TRANSFORMERS_AVAILABLE:
        st.sidebar.error("Transformers library not available. Start Streamlit with:\n`python3 -m streamlit run demo/app.py --server.fileWatcherType none`")
        return

    popular_stocks = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "BRK-B", "JPM", "V",
        "WMT", "JNJ", "PG", "MA", "HD"
    ]
    ticker = st.sidebar.selectbox("Ticker", options=popular_stocks, index=0)

    forecast_horizon = 5
    st.sidebar.caption("Horizon: 5 trading days (fixed by model architecture)")

    try:
        yaml_config = read_yaml(CONFIG_PATH)
        tft_config = read_yaml(TFT_CONFIG_PATH) if os.path.exists(TFT_CONFIG_PATH) else {}
        config = {**tft_config, **yaml_config}
        config.setdefault('model', {})
        config['model']['output_size'] = forecast_horizon
        config['num_articles'] = 4
    except Exception as e:
        st.error(f"Error loading configs: {e}")
        return

    with st.spinner(f"Loading {model_choice}..."):
        is_multimodal = "Multimodal" in model_choice
        try:
            model, model_loaded, device, experiment_path = load_model(config, model_choice)
        except RuntimeError as e:
            # TFT model definitions are not committed with this checkout; let
            # users switch to LSTM/TSMixer without seeing a startup warning.
            st.info(str(e))
            return
        model_type = model_choice.split(" ")[0]

    if not model_loaded:
        st.sidebar.warning(f"No pre-trained weights for {model_type}. Using untrained baseline.")
    else:
        st.sidebar.success(f"{model_type} weights loaded.")

    run_clicked = st.sidebar.button("Run Forecast")

    if not run_clicked:
        st.markdown(
            f"<div class='terminal-meta' style='padding:24px 4px;'>READY · Configure parameters in the sidebar and press <span style='color:{ACCENT_ORANGE}'>RUN FORECAST</span>.</div>",
            unsafe_allow_html=True,
        )
        return

    with st.spinner(f"Fetching live data & indicators for {ticker}..."):
        df = fetch_stock_data(ticker, days=150)

    if df.empty:
        st.error(f"No data found for ticker {ticker}.")
        return

    if "LSTM" in model_choice:
        try:
            run_lstm_backtest(model, experiment_path, device, df, ticker)
        except Exception as e:
            st.error(f"LSTM forecast failed: {e}")
        return

    if "TSMixer" in model_choice:
        try:
            run_tsmixer_backtest(model, experiment_path, device, df, ticker)
        except Exception as e:
            st.error(f"TSMixer forecast failed: {e}")
        return

    tokenized_news = None
    attention_mask = None
    news_articles = []
    if is_multimodal:
        with st.spinner(f"Fetching live news articles for {ticker}..."):
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

    processed_df = preprocess_data(df)

    seq_len = config.get('HISTORY_WINDOW_SIZE', 60)
    total_required = seq_len + forecast_horizon

    if len(processed_df) < total_required:
        st.error("Not enough data to perform backtest.")
        return

    model_input_data = processed_df.iloc[-(total_required):-forecast_horizon].copy()
    ground_truth_data = processed_df.iloc[-forecast_horizon:].copy()

    features = ['open', 'high', 'low', 'close', 'volume',
                'bb_upper', 'bb_middle', 'bb_lower', 'rsi',
                'macd', 'macd_signal', 'macd_histogram']

    X_raw = model_input_data[features].values

    scaler_path = os.path.join(experiment_path, 'preprocessed_data.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            pp = pickle.load(f)
            scaler = pp.get('scaler')

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(processed_df[features].values)

    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with st.spinner("Generating Forecast..."):
        with torch.no_grad():
            if is_multimodal:
                # Live unseen tickers/sectors map to id 0 since the embedding
                # tables only know training-set ids.
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

    dates = model_input_data['date'].tolist()
    future_dates = ground_truth_data['date'].tolist()
    true_future_prices = ground_truth_data['close'].values

    current_price = float(model_input_data['close'].iloc[-1])
    predicted_end = float(preds_raw[-1])
    actual_end = float(true_future_prices[-1])
    error_pct = abs(predicted_end - actual_end) / actual_end * 100
    actual_change = (actual_end - current_price) / current_price * 100
    pred_change = (predicted_end - current_price) / current_price * 100

    render_terminal_header(ticker, model_type)

    last_close_date = fmt_date(dates[-1])
    forecast_end_date = fmt_date(future_dates[-1])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(f"Last Close  {last_close_date}", f"${current_price:,.2f}")
    with k2:
        st.metric(f"Actual  {forecast_end_date}", f"${actual_end:,.2f}", delta=f"{actual_change:+.2f}%")
    with k3:
        st.metric(f"Predicted  {forecast_end_date}", f"${predicted_end:,.2f}", delta=f"{pred_change:+.2f}%")
    with k4:
        st.metric("Error", f"{error_pct:.2f}%", delta=f"{error_pct:.2f}%", delta_color="inverse")

    plot_days = 30
    past_dates = dates[-plot_days:]
    past_closes = model_input_data['close'].values[-plot_days:]

    tabs = ["Forecast", "Data"]
    if is_multimodal and news_articles:
        tabs.append("News Context")
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        fig = build_backtest_figure(past_dates, past_closes, future_dates, true_future_prices, preds_raw)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab_objs[1]:
        forecast_df = pd.DataFrame({
            "Date": [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in future_dates],
            "Actual": [f"${p:,.2f}" for p in true_future_prices],
            "Predicted": [f"${p:,.2f}" for p in preds_raw],
            "Abs Error": [f"${abs(a-p):,.2f}" for a, p in zip(true_future_prices, preds_raw)],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True, height=240)

    if is_multimodal and news_articles:
        with tab_objs[2]:
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
