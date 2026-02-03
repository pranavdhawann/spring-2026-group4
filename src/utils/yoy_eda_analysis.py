import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("S&P500_time_series")
OUTPUT_DIR = Path("analysis_results")
START_YEAR = 2000
END_YEAR = 2025

def load_stock_data(ticker_file):
    try:
        df = pd.read_csv(ticker_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        df['Date'] = df['Date'].dt.tz_localize(None)
        df = df.dropna(subset=['Date']).sort_values('Date')
        df['Year'] = df['Date'].dt.year
        return df
    except Exception as e:
        return None

def calculate_yearly_stock_metrics(df, ticker):
    yearly_data = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        year_df = df[df['Year'] == year]
        
        if len(year_df) == 0:
            continue
        
        expected_days = 252
        if year == 2025:
            expected_days = 59
        
        start_price = year_df.iloc[0]['Close']
        end_price = year_df.iloc[-1]['Close']
        
        trading_days = len(year_df)
        missing_days = expected_days - trading_days
        data_completeness = (trading_days / expected_days) * 100
        
        min_price = year_df['Close'].min()
        max_price = year_df['Close'].max()
        avg_price = year_df['Close'].mean()
        price_range_pct = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
        
        yoy_return = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
        
        daily_returns = year_df['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        avg_volume = year_df['Volume'].mean()
        
        decade = (year // 10) * 10
        
        metrics = {
            'Ticker': ticker,
            'Year': year,
            'Trading_Days': trading_days,
            'Start_Price': start_price,
            'End_Price': end_price,
            'YoY_Return_%': yoy_return,
            'Avg_Price': avg_price,
            'Min_Price': min_price,
            'Max_Price': max_price,
            'Price_Range_%': price_range_pct,
            'Volatility_%': volatility,
            'Avg_Volume': avg_volume,
            'Missing_Days': missing_days,
            'Data_Completeness_%': data_completeness,
            'Decade': decade
        }
        
        yearly_data.append(metrics)
    
    return pd.DataFrame(yearly_data)

def calculate_market_summary(all_data):
    summary = all_data.groupby('Year').agg({
        'Ticker': 'count',
        'YoY_Return_%': ['mean', 'median', 'std'],
        'Volatility_%': ['mean', 'median'],
        'Data_Completeness_%': 'mean'
    })
    
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    summary = summary.rename(columns={'Ticker_count': 'count'})
    
    return summary

def main():
    print("=" * 70)
    print("S&P500 Year-over-Year EDA Analysis (ORIGINAL)")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"\nFound {len(csv_files)} stock CSV files")
    
    all_yearly_data = []
    
    print("\nProcessing stocks for YoY analysis...")
    for csv_file in tqdm(csv_files, desc="Analyzing stocks"):
        ticker = csv_file.stem.upper()
        
        df = load_stock_data(csv_file)
        if df is None or len(df) == 0:
            continue
        
        yearly_metrics = calculate_yearly_stock_metrics(df, ticker)
        if len(yearly_metrics) > 0:
            all_yearly_data.append(yearly_metrics)
    
    print("\nCombining all yearly data...")
    yoy_analysis_full = pd.concat(all_yearly_data, ignore_index=True)
    
    print("Calculating market-wide summary...")
    yearly_summary_full = calculate_market_summary(yoy_analysis_full)
    
    print("\nSaving results...")
    
    yoy_output = OUTPUT_DIR / "yoy_analysis_full.csv"
    yoy_analysis_full.to_csv(yoy_output, index=False)
    print(f"Saved: {yoy_output}")
    print(f"  Records: {len(yoy_analysis_full):,}")
    
    summary_output = OUTPUT_DIR / "yearly_summary_full.csv"
    yearly_summary_full.to_csv(summary_output)
    print(f"Saved: {summary_output}")
    print(f"  Years: {len(yearly_summary_full)}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total stocks analyzed: {yoy_analysis_full['Ticker'].nunique()}")
    print(f"Years covered: {START_YEAR} - {END_YEAR}")
    print(f"Total stock-year records: {len(yoy_analysis_full):,}")
    print(f"\nAverage annual return: {yoy_analysis_full['YoY_Return_%'].mean():.2f}%")
    print(f"Average volatility: {yoy_analysis_full['Volatility_%'].mean():.2f}%")
    print(f"Average data completeness: {yoy_analysis_full['Data_Completeness_%'].mean():.2f}%")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
