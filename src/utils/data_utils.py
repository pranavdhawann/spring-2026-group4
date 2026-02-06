import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from utils import read_yaml

# Load configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"
CONFIG = read_yaml(str(CONFIG_PATH))

# Set paths from config
DATA_DIR = PROJECT_ROOT / CONFIG['DATA_FOLDER'].lstrip('../')
STOCK_DATA_DIR = PROJECT_ROOT / CONFIG['STOCK_DATA_FOLDER'].lstrip('../')
ANALYSIS_DIR = PROJECT_ROOT / CONFIG['ANALYSIS_FOLDER'].lstrip('../')
NEWS_DIR = PROJECT_ROOT / CONFIG['NEWS_FOLDER'].lstrip('../')
DATA_DICT_PATH = PROJECT_ROOT / CONFIG['DATA_DICTIONARY'].lstrip('../')


def get_project_paths():
    return {
        'project_root': PROJECT_ROOT,
        'data': DATA_DIR,
        'stock_data': STOCK_DATA_DIR,
        'analysis': ANALYSIS_DIR,
        'news': NEWS_DIR,
        'data_dictionary': DATA_DICT_PATH
    }


def load_stock_csv(ticker: str, start_year: Optional[int] = None) -> pd.DataFrame:
    csv_file = STOCK_DATA_DIR / f"{ticker.lower()}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Stock data not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.dropna(subset=['Date']).sort_values('Date')
    
    if start_year:
        df = df[df['Date'].dt.year >= start_year]
    
    return df


def load_yoy_analysis(start_year: Optional[int] = None) -> pd.DataFrame:
    yoy_file = ANALYSIS_DIR / 'yoy_analysis_full.csv'
    df = pd.read_csv(yoy_file)
    
    if start_year:
        df = df[df['Year'] >= start_year]
    
    return df


def calculate_completeness_score(completeness_pct: pd.Series) -> pd.Series:
    return completeness_pct / 100


def calculate_years_score(years: pd.Series) -> pd.Series:
    return years / years.max()


def calculate_quality_score(
    completeness_pct: pd.Series,
    years: pd.Series,
    completeness_weight: float = 0.7,
    years_weight: float = 0.3
) -> pd.Series:
    completeness_score = calculate_completeness_score(completeness_pct)
    years_score = calculate_years_score(years)
    
    return (completeness_score * completeness_weight + 
            years_score * years_weight)


def aggregate_ticker_metrics(
    df: pd.DataFrame,
    groupby_col: str = 'Ticker'
) -> pd.DataFrame:
    metrics = df.groupby(groupby_col).agg({
        'Year': 'count',
        'Data_Completeness_%': 'mean',
        'Volatility_%': 'mean',
        'YoY_Return_%': 'mean',
        'Avg_Price': 'mean'
    }).rename(columns={
        'Year': 'Years_of_Data',
        'Data_Completeness_%': 'Avg_Completeness_%',
        'Volatility_%': 'Avg_Volatility_%',
        'YoY_Return_%': 'Avg_Return_%',
        'Avg_Price': 'Avg_Price'
    })
    
    return metrics


def save_ticker_list(tickers: List[str], output_file: Path) -> None:
    with open(output_file, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")


def load_ticker_list(input_file: Path) -> List[str]:
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_col: str = 'Date'
) -> pd.DataFrame:
    df_filtered = df.copy()
    
    if start_date:
        df_filtered = df_filtered[df_filtered[date_col] >= start_date]
    
    if end_date:
        df_filtered = df_filtered[df_filtered[date_col] <= end_date]
    
    return df_filtered


def get_data_summary(df: pd.DataFrame, name: str = "Dataset") -> dict:
    summary = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    if 'Ticker' in df.columns:
        summary['unique_tickers'] = df['Ticker'].nunique()
    
    if 'Date' in df.columns:
        summary['date_range'] = (df['Date'].min(), df['Date'].max())
    
    return summary

