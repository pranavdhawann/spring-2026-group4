import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict


root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.utils import read_yaml, working_directory_to_src, load_stock_csv

working_directory_to_src()

#Fallback
if Path.cwd().name != 'src' and (Path.cwd() / 'src').exists():
    os.chdir('src')


def _get_config() -> dict:
    working_directory_to_src()
    src_path = Path.cwd()
    config_path = src_path / "config.yaml"
    return read_yaml(str(config_path))


def get_data_paths() -> Dict[str, Path]:
    config = _get_config()
    src_path = Path.cwd()
    project_root = src_path.parent
    
    return {
        'data_dir': project_root / config['DATA_FOLDER'].lstrip('../'),
        'stock_data_dir': project_root / config['STOCK_DATA_FOLDER'].lstrip('../'),
        'analysis_dir': project_root / config['ANALYSIS_FOLDER'].lstrip('../'),
        'news_dir': project_root / config['NEWS_FOLDER'].lstrip('../'),
        'data_dict_path': project_root / config['DATA_DICTIONARY'].lstrip('../'),
        'stock_score_news_path': project_root / config['STOCK_SCORE_NEWS'].lstrip('../')
    }


def save_ticker_list(tickers: List[str], output_file: Path) -> None:
    with open(output_file, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")


def load_ticker_list(input_file: Path) -> List[str]:
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


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


def load_yoy_analysis(analysis_dir: Path, start_year: Optional[int] = None) -> pd.DataFrame:
    yoy_file = analysis_dir / 'yoy_analysis_full.csv'
    df = pd.read_csv(yoy_file)
    
    if start_year:
        df = df[df['Year'] >= start_year]
    
    return df


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
