import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import read_yaml, load_stock_csv

START_YEAR = 2000
END_YEAR = 2025

config = read_yaml("config/config.yaml")


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
        min_price = year_df['Close'].min()
        max_price = year_df['Close'].max()
        avg_price = year_df['Close'].mean()
        
        # Completeness
        trading_days = len(year_df)
        missing_days = expected_days - trading_days
        data_completeness = (trading_days / expected_days) * 100
        
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


def calculate_scores_from_yoy(yoy_df, start_year=2009):
    from cookbooks.EDA.eda_utils import (
        aggregate_ticker_metrics,
        calculate_quality_score
    )
    
    if start_year:
        yoy_filtered = yoy_df[yoy_df['Year'] >= start_year]
    else:
        yoy_filtered = yoy_df
    
    print(f"Filtered to {start_year}+: {len(yoy_filtered):,} records")
    print(f"Unique tickers: {yoy_filtered['Ticker'].nunique()}")
    
    print("Calculating quality scores per ticker...")
    ticker_metrics = aggregate_ticker_metrics(yoy_filtered)
    
    # Calculate quality score
    # Quality_Score = 0.7 * (Avg_Completeness/100) + 0.3 * (Years/MaxYears)
    ticker_metrics['Quality_Score'] = calculate_quality_score(
        ticker_metrics['Avg_Completeness_%'],
        ticker_metrics['Years_of_Data']
    )
    
    ranked_stocks = ticker_metrics.sort_values('Quality_Score', ascending=False)
    
    return ranked_stocks


def main():

    print("STOCK TIME SERIES QUALITY SCORING")
    
    stock_data_folder = Path(config["TIMES_SERIES_FOLDER"])
    output_multimodal = Path(config["STOCK_SCORE_TIMESERIES"]).parent
    output_multimodal.mkdir(exist_ok=True, parents=True)
    
    
    if not stock_data_folder.exists():
        print(f"Error: Stock data folder not found: {stock_data_folder}")
        return
    
    csv_files = sorted(stock_data_folder.glob("*.csv"))
    print(f"\nFound {len(csv_files)} stock CSV files")
    

    print("STEP 1: Calculating Yearly Metrics (YoY Analysis)")

    
    all_yearly_data = []
    
    print("\nProcessing stocks for YoY analysis...")
    for csv_file in tqdm(csv_files, desc="Analyzing stocks"):
        ticker = csv_file.stem.upper()
        
        try:
            df = load_stock_csv(ticker, stock_data_folder, start_year=None)
            if df is None or len(df) == 0:
                continue
            
            df['Year'] = df['Date'].dt.year
            
            yearly_metrics = calculate_yearly_stock_metrics(df, ticker)
            if len(yearly_metrics) > 0:
                all_yearly_data.append(yearly_metrics)
                
        except Exception:
            continue
    
    if not all_yearly_data:
        print("ERROR: No stocks were successfully processed!")
        return
    
    print("\nCombining all yearly data...")
    yoy_analysis_full = pd.concat(all_yearly_data, ignore_index=True)
    
    yoy_output = output_multimodal / "yoy_analysis_full.csv"
    yoy_analysis_full.to_csv(yoy_output, index=False)
    print(f"\nSaved YoY analysis: {yoy_output}")
    print(f"   Records: {len(yoy_analysis_full):,}")
    print(f"   Stocks: {yoy_analysis_full['Ticker'].nunique()}")
    
    print("\n" + "=" * 70)
    print("STEP 2: Calculating Quality Scores (2009-2025)")
    print("=" * 70)
    
    ranked_stocks = calculate_scores_from_yoy(yoy_analysis_full, start_year=2009)
    
    print("\n" + "=" * 60)
    print("ALL STOCKS RANKED BY QUALITY")
    print("=" * 60)
    print(f"\nTotal stocks ranked: {len(ranked_stocks)}")
    print(f"\nQuality Score Statistics:")
    print(f"  Mean: {ranked_stocks['Quality_Score'].mean():.4f}")
    print(f"  Min: {ranked_stocks['Quality_Score'].min():.4f}")
    print(f"  Max: {ranked_stocks['Quality_Score'].max():.4f}")
    
    print(f"\nData Coverage Statistics:")
    print(f"  Avg years of data: {ranked_stocks['Years_of_Data'].mean():.1f}")
    print(f"  Avg completeness: {ranked_stocks['Avg_Completeness_%'].mean():.2f}%")
    print(f"  Avg volatility: {ranked_stocks['Avg_Volatility_%'].mean():.2f}%")
    print(f"  Avg annual return: {ranked_stocks['Avg_Return_%'].mean():.2f}%")
    
    print(f"\nTop 20 stocks by quality score:")
    print(ranked_stocks[['Quality_Score', 'Years_of_Data', 'Avg_Completeness_%', 'Avg_Volatility_%']].head(20))
    
    print("\n" + "=" * 70)
    print("STEP 3: Saving Ranked Stocks")
    print("=" * 70)
    
    all_stocks_csv = output_multimodal / 'all_stocks_ranked.csv'
    ranked_stocks.to_csv(all_stocks_csv)
    print(f"\nRanked stocks saved to: {all_stocks_csv}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total stocks analyzed: {yoy_analysis_full['Ticker'].nunique()}")
    print(f"Stocks ranked (2009+): {len(ranked_stocks)}")
    print(f"Years covered: {START_YEAR} - {END_YEAR}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    print("\nOutput files:")
    print(f"  1. {yoy_output}")
    print(f"  2. {all_stocks_csv}")


if __name__ == "__main__":
    main()