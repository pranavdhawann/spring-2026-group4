"""
utils_gnn.py — Calendar utilities, reproducibility helpers, date alignment, trading day computations.
"""

import logging
import os
import platform
import random
import sys
from datetime import datetime
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join("config", "config_gnn.yaml")
DEFAULT_CHECKPOINT_DIR = os.path.join("tgnn", "checkpoints")
DEFAULT_BEST_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "best.pt")
DEFAULT_RESULTS_DIR = os.path.join("tgnn", "results")
DEFAULT_LOG_DIR = os.path.join("tgnn", "logs", "runs")
DEFAULT_TENSORBOARD_DIR = os.path.join("tgnn", "logs", "tensorboard")


def resolve_data_path(
    data_dir: str,
    configured_path: Optional[str],
    default_path: str,
    *,
    kind: str = "directory",
    aliases: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Resolve a data path relative to `data_dir`.

    Tries the configured path first, then the default, then any aliases. If no
    existing path is found, returns the configured/default path joined to
    `data_dir` so callers can decide whether to create it or fail later.
    """
    rel_candidates: List[str] = []
    for rel_path in [configured_path, default_path, *(aliases or [])]:
        if rel_path and rel_path not in rel_candidates:
            rel_candidates.append(rel_path)

    exists_fn = os.path.isfile if kind == "file" else os.path.isdir
    checked_paths: List[str] = []

    for rel_path in rel_candidates:
        abs_path = os.path.join(data_dir, rel_path)
        checked_paths.append(os.path.abspath(abs_path))
        if exists_fn(abs_path):
            if configured_path and rel_path != configured_path:
                logger.warning(
                    "Configured %s path not found: %s | using fallback: %s",
                    kind,
                    os.path.abspath(os.path.join(data_dir, configured_path)),
                    os.path.abspath(abs_path),
                )
            return rel_path, abs_path

    chosen_rel = configured_path or default_path
    chosen_abs = os.path.join(data_dir, chosen_rel)
    if checked_paths:
        logger.warning(
            "Unable to resolve %s path. Checked: %s",
            kind,
            ", ".join(checked_paths),
        )
    return chosen_rel, chosen_abs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _resolve_log_level(level_name, default=logging.INFO):
    if isinstance(level_name, int):
        return level_name
    if not level_name:
        return default
    return getattr(logging, str(level_name).upper(), default)


def setup_logging(config: Optional[dict] = None, command_name: str = "app", config_path: Optional[str] = None,
                  args=None) -> str:
    """
    Configure project-wide logging with both console and per-run file handlers.

    Returns:
        Absolute path to the log file for the current run.
    """
    log_cfg = (config or {}).get("logging", {})
    root_dir = log_cfg.get("dir", DEFAULT_LOG_DIR)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    command_slug = command_name.replace(" ", "_").replace("/", "_")
    log_dir = os.path.abspath(os.path.join(root_dir, command_slug))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.log")

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    root_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_resolve_log_level(log_cfg.get("console_level", log_cfg.get("level", "INFO"))))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(_resolve_log_level(log_cfg.get("file_level", "DEBUG"), logging.DEBUG))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if log_cfg.get("capture_warnings", True):
        logging.captureWarnings(True)

    logging.getLogger("matplotlib").setLevel(_resolve_log_level(log_cfg.get("matplotlib_level", "WARNING")))
    logging.getLogger("urllib3").setLevel(_resolve_log_level(log_cfg.get("urllib3_level", "WARNING")))

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("uncaught").exception(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _log_uncaught_exception

    logger.info("=" * 80)
    logger.info("Starting %s", command_name)
    logger.info("Log file: %s", log_path)
    if config_path:
        logger.info("Config path: %s", os.path.abspath(config_path))
    if args is not None:
        try:
            logger.info("CLI args: %s", vars(args))
        except TypeError:
            logger.info("CLI args: %s", args)

    return log_path


def log_runtime_context(command_name: str, config: Optional[dict] = None, extra: Optional[dict] = None):
    """Log environment and configuration details for the current run."""
    logger.info("Command: %s", command_name)
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Platform: %s", platform.platform())
    logger.info("Python: %s", sys.version.replace("\n", " "))

    try:
        import torch
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        logger.info(
            "Torch runtime: version=%s | cuda_available=%s | device=%s",
            torch.__version__,
            torch.cuda.is_available(),
            cuda_name,
        )
    except Exception as exc:
        logger.warning("Unable to inspect torch runtime: %s", exc)

    if config:
        data_cfg = config.get("data", {})
        train_cfg = config.get("training", {})
        log_cfg = config.get("logging", {})
        logger.info(
            "Config summary | data_dir=%s | window=%s | horizon=%s | backend=%s | project=%s",
            data_cfg.get("data_dir", "data"),
            data_cfg.get("window_size"),
            data_cfg.get("horizon"),
            log_cfg.get("backend"),
            log_cfg.get("project"),
        )
        logger.info(
            "Training summary | epochs=%s | lr=%s | batch_size=%s | grad_accum=%s | mixed_precision=%s",
            train_cfg.get("max_epochs"),
            train_cfg.get("lr"),
            train_cfg.get("batch_size", 1),
            train_cfg.get("grad_accumulation_steps"),
            train_cfg.get("mixed_precision"),
        )

    if extra:
        for key, value in extra.items():
            logger.info("%s: %s", key, value)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}, deterministic={deterministic}")


# ---------------------------------------------------------------------------
# Trading Calendar
# ---------------------------------------------------------------------------

def build_trading_calendar(
    start_date: str = "2000-01-01",
    end_date: str = "2026-12-31",
) -> pd.DatetimeIndex:
    """
    Build a US stock market trading calendar using pandas_market_calendars.
    Falls back to pandas business day calendar with known US holidays if
    pandas_market_calendars is unavailable.
    """
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index
        logger.info(f"Built NYSE trading calendar: {len(trading_days)} days "
                     f"({trading_days[0].date()} to {trading_days[-1].date()})")
        return trading_days
    except ImportError:
        logger.warning("pandas_market_calendars not available; using pd.bdate_range as fallback")
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
        return trading_days


def get_next_trading_day(date: pd.Timestamp, trading_calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the next trading day on or after `date`."""
    mask = trading_calendar >= date
    if mask.any():
        return trading_calendar[mask][0]
    raise ValueError(f"No trading day found on or after {date}")


def get_prev_trading_day(date: pd.Timestamp, trading_calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the most recent trading day on or before `date`."""
    mask = trading_calendar <= date
    if mask.any():
        return trading_calendar[mask][-1]
    raise ValueError(f"No trading day found on or before {date}")


def build_master_calendar(ticker_dfs: dict) -> pd.DatetimeIndex:
    """
    Build a master calendar from the union of all tickers' trading dates.
    
    Args:
        ticker_dfs: dict mapping ticker -> DataFrame with 'date' column
    
    Returns:
        Sorted DatetimeIndex of all unique trading dates
    """
    all_dates = set()
    for ticker, df in ticker_dfs.items():
        dates = pd.to_datetime(df["date"])
        all_dates.update(dates.tolist())
    master = pd.DatetimeIndex(sorted(all_dates))
    logger.info(f"Master calendar: {len(master)} trading days "
                 f"({master[0].date()} to {master[-1].date()}) "
                 f"from {len(ticker_dfs)} tickers")
    return master


# ---------------------------------------------------------------------------
# Date / News Alignment
# ---------------------------------------------------------------------------

def assign_news_to_trading_day(
    article_date: pd.Timestamp,
    trading_calendar: pd.DatetimeIndex,
    cutoff_hour: int = 16,
    cutoff_minute: int = 0,
) -> pd.Timestamp:
    """
    Assign a news article to the correct trading day based on publication time.
    
    Rules:
        - Weekday before cutoff (default 16:00 ET) → current trading day
        - Weekday after cutoff → next trading day
        - Weekend / holiday → next trading day
        - Pre-market (before 9:30am) → current trading day
    
    Args:
        article_date: Publication datetime (assumed ET timezone)
        trading_calendar: Sorted DatetimeIndex of trading days
        cutoff_hour: Hour of cutoff time (ET)
        cutoff_minute: Minute of cutoff time (ET)
    
    Returns:
        Trading day pd.Timestamp to which this article should be assigned
    """
    date_only = pd.Timestamp(article_date.date())
    
    # Check if it's a trading day
    is_trading_day = date_only in trading_calendar
    
    if is_trading_day:
        # Check if article is after cutoff
        if hasattr(article_date, 'hour'):
            article_hour = article_date.hour
            article_minute = article_date.minute
        else:
            # Date only — assume before cutoff
            return date_only
        
        if article_hour > cutoff_hour or (article_hour == cutoff_hour and article_minute >= cutoff_minute):
            # After cutoff → next trading day
            return get_next_trading_day(date_only + pd.Timedelta(days=1), trading_calendar)
        else:
            return date_only
    else:
        # Not a trading day → next trading day
        return get_next_trading_day(date_only, trading_calendar)


# ---------------------------------------------------------------------------
# Feature Computation Helpers
# ---------------------------------------------------------------------------

def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute log returns: log(close_t / close_{t-1})."""
    close = pd.to_numeric(close, errors="coerce")
    prev_close = close.shift(1)
    valid = (close > 0) & (prev_close > 0)

    log_returns = pd.Series(np.nan, index=close.index, dtype=float)
    log_returns.loc[valid] = np.log(close.loc[valid] / prev_close.loc[valid])
    return log_returns


def safe_log1p(series: pd.Series) -> pd.Series:
    """Compute log(1 + x) after clipping negative values to zero."""
    series = pd.to_numeric(series, errors="coerce").clip(lower=0)
    return np.log1p(series)


def mad_zscore(series: pd.Series) -> pd.Series:
    """
    Compute MAD-based z-scores (robust to outliers).
    z_mad = 0.6745 * (x - median(x)) / MAD
    where MAD = median(|x - median(x)|)
    """
    median_val = series.median()
    mad = (series - median_val).abs().median()
    if mad < 1e-10:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median_val) / mad


def clip_log_returns(log_returns: pd.Series, sigma: float = 5.0) -> pd.Series:
    """
    Clip log returns at ±sigma using an **expanding-window** MAD-based z-score.

    FIX E6: the previous implementation computed MAD over the full series
    (including future val/test data), leaking distribution information into
    training features.  The expanding window ensures that clipping thresholds
    at each date depend only on data available up to that date.
    """
    clipped = log_returns.copy()
    valid = log_returns.dropna()
    if len(valid) < 2:
        return clipped

    min_periods = 60  # need enough history for stable MAD

    median_expanding = valid.expanding(min_periods=min_periods).median()
    mad_expanding = (
        (valid - median_expanding)
        .abs()
        .expanding(min_periods=min_periods)
        .median()
    )

    # 0.6745 converts MAD to σ-equivalent scale
    z_expanding = 0.6745 * (valid - median_expanding) / (mad_expanding + 1e-10)
    outlier_mask = z_expanding.abs() > sigma

    n_clipped = int(outlier_mask.sum())
    if n_clipped > 0:
        logger.debug(f"Clipped {n_clipped} log returns at ±{sigma}σ (MAD-based)")

    # For each outlier, clip to the nearest ±sigma boundary at that point
    upper_bound = median_expanding + sigma * (mad_expanding / 0.6745)
    lower_bound = median_expanding - sigma * (mad_expanding / 0.6745)
    clipped.loc[valid.index] = valid.clip(lower=lower_bound, upper=upper_bound)

    return clipped


def expanding_zscore(
    series: pd.Series,
    min_periods: int = 60,
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
) -> pd.Series:
    """
    Expanding window z-score normalization (prevents leakage).
    Uses global statistics as fallback for periods with < min_periods observations.
    """
    exp_mean = series.expanding(min_periods=min_periods).mean()
    exp_std = series.expanding(min_periods=min_periods).std()
    
    # Replace NaN periods with global stats if available
    if global_mean is not None and global_std is not None:
        exp_mean = exp_mean.fillna(global_mean)
        exp_std = exp_std.fillna(global_std)
    
    # Avoid division by zero
    exp_std = exp_std.replace(0, 1e-8)
    
    return (series - exp_mean) / exp_std


# ---------------------------------------------------------------------------
# Data Split
# ---------------------------------------------------------------------------

def temporal_train_val_test_split(
    master_calendar: pd.DatetimeIndex,
    val_days: int = 45,
    test_days: int = 45,
    purge_days: int = 5,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Temporal split with purge gaps.
    
    |◄──── Train ─────►|◄─purge─►|◄── Val ──►|◄─purge─►|◄── Test ──►|
    
    Returns:
        (train_dates, val_dates, test_dates)
    """
    n = len(master_calendar)
    
    test_start = n - test_days
    val_end = test_start - purge_days
    val_start = val_end - val_days
    train_end = val_start - purge_days
    
    if train_end <= 0:
        raise ValueError(
            f"Not enough data for split: {n} total days, need at least "
            f"{val_days + test_days + 2 * purge_days + 1} days"
        )
    
    train_dates = master_calendar[:train_end]
    val_dates = master_calendar[val_start:val_end]
    test_dates = master_calendar[test_start:]
    
    logger.info(f"Train: {len(train_dates)} days ({train_dates[0].date()} to {train_dates[-1].date()})")
    logger.info(f"Val:   {len(val_dates)} days ({val_dates[0].date()} to {val_dates[-1].date()})")
    logger.info(f"Test:  {len(test_dates)} days ({test_dates[0].date()} to {test_dates[-1].date()})")
    
    return train_dates, val_dates, test_dates


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_metric: float,
    path: str,
    scheduler=None,
    scaler=None,
    extra: dict = None,
):
    """Save model checkpoint."""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_metric": val_metric,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if extra:
        state.update(extra)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path} (epoch={epoch}, val_metric={val_metric:.6f})")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None):
    """Load model checkpoint.

    FIX E9: PyTorch 2.6+ switched the default of ``weights_only`` in
    ``torch.load`` to ``True``, which refuses to unpickle arbitrary Python
    objects (including numpy scalars in our ``extra`` dict).  Our own
    checkpoints also contain the config dict, so we must explicitly pass
    ``weights_only=False``.  This is safe because we only ever load our own
    files.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    logger.info(f"Loaded checkpoint from {path} (epoch={state.get('epoch', '?')})")
    return state


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def count_parameters(model) -> int:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable")
    return trainable


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device
