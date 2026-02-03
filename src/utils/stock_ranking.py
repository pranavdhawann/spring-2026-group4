import pandas as pd
from data_utils import (
    load_yoy_analysis,
    aggregate_ticker_metrics,
    calculate_quality_score,
    save_ticker_list,
    get_project_paths
)

def rank_all_stocks(ticker_metrics):
    print("Ranking all stocks by quality score...")
    ranked_stocks = ticker_metrics.sort_values('Quality_Score', ascending=False)
    return ranked_stocks

def display_results(ranked_stocks):
    print("="*60)
    print("ALL STOCKS RANKED BY QUALITY")
    print("="*60)
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

def save_results(ranked_stocks):
    paths = get_project_paths()
    output_dir = paths['analysis']
    
    all_stocks_csv = output_dir / 'all_stocks_ranked.csv'
    ranked_stocks.to_csv(all_stocks_csv)
    print(f"\n{'='*60}")
    print(f"All ranked stocks saved to: {all_stocks_csv}")
    print(f"{'='*60}")
    
    all_tickers_txt = output_dir / 'all_stocks_ranked.txt'
    save_ticker_list(ranked_stocks.index.tolist(), all_tickers_txt)
    print(f"Ticker list saved to: {all_tickers_txt}")

def main():
    print("="*60)
    print("STOCK QUALITY RANKING (2009-2025)")
    print("="*60 + "\n")
    
    yoy_df = load_yoy_analysis(start_year=2009)
    print(f"Loaded {len(yoy_df):,} records (2009-2025)")
    print(f"Unique tickers: {yoy_df['Ticker'].nunique()}\n")
    
    print("Calculating quality scores per ticker...")
    ticker_metrics = aggregate_ticker_metrics(yoy_df)
    
    ticker_metrics['Quality_Score'] = calculate_quality_score(
        ticker_metrics['Avg_Completeness_%'],
        ticker_metrics['Years_of_Data']
    )
    
    ranked_stocks = rank_all_stocks(ticker_metrics)
    display_results(ranked_stocks)
    save_results(ranked_stocks)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
