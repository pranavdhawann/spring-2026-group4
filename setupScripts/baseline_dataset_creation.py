import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from src.utils.utils import read_yaml



def load_financial_jsonl(filepath):
    """Load JSON or JSONL file"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
        
        if not records:
            f.seek(0)
            try:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
            except:
                pass
    
    return records


def extract_financial_features(financial_data):
    """Extract financial features from JSON data into DataFrame"""
    all_rows = []
    
    for statement_type, records in financial_data.items():
        print(f"  Processing {statement_type}...")
        
        for company_record in records:
            cik = company_record.get('cik', '')
            company_name = company_record.get('company_name', '')
            
            for filing in company_record.get('filings', []):
                filing_date = filing.get('filing_date', '')
                form = filing.get('form', '')
                year = filing.get('year', '')
                
                row = {
                    'cik': cik,
                    'company_name': company_name,
                    'statement_type': statement_type,
                    'filing_date': filing_date,
                    'form': form,
                    'year': year
                }
                
                facts = filing.get('facts', {})
                
                for taxonomy, metrics in facts.items():
                    for metric_name, metric_data in metrics.items():
                        col_name = f"{taxonomy}_{metric_name}"
                        
                        if 'units' not in metric_data:
                            continue
                        
                        units = metric_data['units']
                        
                        for unit_type in ['USD', 'shares', 'pure', 'USD/shares']:
                            if unit_type not in units:
                                continue
                            
                            values = units[unit_type]
                            if not values or not isinstance(values, list):
                                continue
                            
                            matched_value = None
                            
                            # Strategy 1: Match by 'filed' date
                            for val_entry in values:
                                if val_entry.get('filed') == filing_date:
                                    matched_value = val_entry.get('val')
                                    break
                            
                            # Strategy 2: Match by form and fiscal year
                            if matched_value is None:
                                for val_entry in values:
                                    if (val_entry.get('form') == form and 
                                        str(val_entry.get('fy')) == str(year)):
                                        matched_value = val_entry.get('val')
                                        break
                            
                            # Strategy 3: Most recent for this filing
                            if matched_value is None:
                                valid_values = [
                                    v for v in values 
                                    if v.get('filed', '') <= filing_date
                                ]
                                if valid_values:
                                    matched_value = valid_values[-1].get('val')
                            
                            if matched_value is not None:
                                row[col_name] = matched_value
                                break
                
                all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    
    if 'filing_date' in df.columns:
        df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    
    print(f"  Extracted shape: {df.shape}")
    return df


def merge_daily_with_financials(ts_data, date_col, financial_df, ticker):
    """Merge daily OHLCV with quarterly financial data"""
    ts_data = ts_data.copy()
    
    fin_ticker = financial_df[financial_df['company_name'].str.lower() == ticker.lower()].copy()
    
    if fin_ticker.empty:
        print(f"  No financial data found for {ticker}")
        return ts_data
    
    ts_data[date_col] = pd.to_datetime(ts_data[date_col], errors='coerce')
    fin_ticker['filing_date'] = pd.to_datetime(fin_ticker['filing_date'], errors='coerce')
    
    if ts_data[date_col].dt.tz is not None:
        ts_data[date_col] = ts_data[date_col].dt.tz_localize(None)
    
    if fin_ticker['filing_date'].dt.tz is not None:
        fin_ticker['filing_date'] = fin_ticker['filing_date'].dt.tz_localize(None)
    
    ts_data = ts_data.dropna(subset=[date_col])
    fin_ticker = fin_ticker.dropna(subset=['filing_date'])
    
    ts_data = ts_data.sort_values(date_col).reset_index(drop=True)
    fin_ticker = fin_ticker.sort_values('filing_date').reset_index(drop=True)
    
    print(f"  Daily records: {len(ts_data):,}, Financial filings: {len(fin_ticker)}")
    
    merged = pd.merge_asof(
        ts_data,
        fin_ticker,
        left_on=date_col,
        right_on='filing_date',
        direction='backward'
    )
    
    fin_cols = [col for col in merged.columns if col.startswith('us-gaap_')]
    has_fin_data = merged[fin_cols].notna().any(axis=1).sum()
    
    print(f"  Merged: {len(merged):,} rows, {has_fin_data:,} with financial data")
    
    return merged


def load_and_process_news(news_file, ticker):
    """Load news JSONL and aggregate by date"""
    print(f"  Loading news data...")
    
    articles = []
    with open(news_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    articles.append(json.loads(line))
                except:
                    pass
    
    news_df = pd.DataFrame(articles)
    print(f"  Loaded {len(news_df)} articles")
    
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
    if news_df['Date'].dt.tz is not None:
        news_df['Date'] = news_df['Date'].dt.tz_localize(None)
    
    news_df = news_df.dropna(subset=['Date'])
    
    news_agg = news_df.groupby('Date').agg(
        news_count=('Article', 'count'),
        news_avg_length=('Article', lambda x: x.str.len().mean()),
        news_articles_list=('Article', lambda x: x.dropna().tolist())
    ).reset_index()
    
    news_agg = news_agg.rename(columns={'Date': 'date'})
    
    print(f"  Aggregated to {len(news_agg)} unique dates")
    
    return news_agg


def merge_with_news(merged_data, news_agg, date_col):
    """Add news features to merged dataset"""
    if news_agg is None or news_agg.empty:
        print("  No news data to merge")
        return merged_data
    
    merged_with_news = merged_data.copy()
    
    merged_with_news[date_col] = pd.to_datetime(merged_with_news[date_col])
    if merged_with_news[date_col].dt.tz is not None:
        merged_with_news[date_col] = merged_with_news[date_col].dt.tz_localize(None)
    
    news_agg['date'] = pd.to_datetime(news_agg['date'])
    if news_agg['date'].dt.tz is not None:
        news_agg['date'] = news_agg['date'].dt.tz_localize(None)
    
    print(f"  Before merge: {merged_with_news.shape}")
    
    final = merged_with_news.merge(
        news_agg,
        left_on=date_col,
        right_on='date',
        how='left'
    )
    
    if 'date' in final.columns and date_col != 'date':
        final = final.drop(columns=['date'])
    
    final['news_count'] = final['news_count'].fillna(0).astype(int)
    final['news_articles_list'] = final['news_articles_list'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    days_with_news = (final['news_count'] > 0).sum()
    print(f"  After merge: {final.shape}, Days with news: {days_with_news}")
    
    return final


def create_windowed_samples(final_merged, ticker, sector_map, history_window=30, forecast_horizon=5, stride=1):
    """Create windowed samples for forecasting"""
    
    ts_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    table_cols = ['cik', 'company_name', 'statement_type', 'filing_date', 'form', 'year',
                  'us-gaap_AccountsPayableCurrent', 'us-gaap_AccountsReceivableNetCurrent',
                  'us-gaap_Assets', 'us-gaap_AssetsCurrent', 'us-gaap_CommonStockValue',
                  'us-gaap_InventoryNet', 'us-gaap_Liabilities', 'us-gaap_LiabilitiesCurrent',
                  'us-gaap_PropertyPlantAndEquipmentNet', 'us-gaap_StockholdersEquity',
                  'us-gaap_CashAndCashEquivalentsPeriodIncreaseDecrease',
                  'us-gaap_NetCashProvidedByUsedInFinancingActivities',
                  'us-gaap_NetCashProvidedByUsedInInvestingActivities',
                  'us-gaap_NetCashProvidedByUsedInOperatingActivities',
                  'us-gaap_PaymentsToAcquirePropertyPlantAndEquipment',
                  'us-gaap_ProceedsFromSaleOfPropertyPlantAndEquipment',
                  'us-gaap_RetainedEarningsAccumulatedDeficit',
                  'us-gaap_StockIssuedDuringPeriodValueShareBasedCompensation',
                  'us-gaap_StockRepurchaseProgramAuthorizedAmount1',
                  'us-gaap_PaymentsRelatedToTaxWithholdingForShareBasedCompensation',
                  'us-gaap_StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1',
                  'us-gaap_DividendsCommonStockCash', 'us-gaap_DividendsPayableAmountPerShare',
                  'us-gaap_StockIssuedDuringPeriodSharesShareBasedCompensation',
                  'srt_StockRepurchaseProgramAuthorizedAmount1']
    
    samples = []
    total = history_window + forecast_horizon
    sector = sector_map.get(ticker, 'Unknown')
    
    for i in range(0, len(final_merged) - total + 1, stride):
        window = final_merged.iloc[i:i + history_window]
        future = final_merged.iloc[i + history_window:i + total]
        
        sample = {
            "dates": window['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "articles": [None if not isinstance(x, list) or len(x) == 0 else x for x in window['news_articles_list']],
            "time_series": [{col.lower(): (None if pd.isna(row[col]) else float(row[col])) 
                             for col in ts_cols} for _, row in window.iterrows()],
            "table_data": [{col: (None if col not in window.columns or pd.isna(row[col]) 
                          else str(row[col]) if isinstance(row[col], pd.Timestamp)
                          else float(row[col]) if isinstance(row[col], (np.integer, np.floating)) 
                          else row[col]) for col in table_cols} for _, row in window.iterrows()],
            "sector": sector,
            "target": [None if pd.isna(x) else float(x) for x in future['Close']]
        }
        samples.append(sample)
    
    return samples


def process_ticker(ticker, config, project_root, sector_map):
    """Process a single ticker"""
    print(f"\n{'='*70}")
    print(f"Processing: {ticker.upper()}")
    print(f"{'='*70}")
    
    FINANCIAL_REPORTS = project_root / config['TABULAR_DATA']
    TIME_SERIES = project_root / config['TIMES_SERIES_FOLDER']
    NEWS_DIR = project_root / config['NEWS_FOLDER']
    OUTPUT_PATH = project_root / config['OUTPUT_PATH_DATA']
    
    ticker_dir = FINANCIAL_REPORTS / ticker
    financial_data = {}
    
    if ticker_dir.exists() and ticker_dir.is_dir():
        print(f"Loading financial data...")
        json_files = list(ticker_dir.glob("*.json*"))
        
        for file in json_files:
            records = load_financial_jsonl(file)
            name = file.stem.lower()
            
            if 'balance' in name:
                key = 'balance_sheet'
            elif 'cash' in name:
                key = 'cash_flow'
            elif 'income' in name:
                key = 'income_statement'
            elif 'equity' in name:
                key = 'equity'
            else:
                key = file.stem
            
            financial_data[key] = records
        
        features_df = extract_financial_features(financial_data)
    else:
        print(f"  Financial directory not found: {ticker_dir}")
        return None
    
    ts_files = list(TIME_SERIES.glob(f"{ticker}.csv"))
    if not ts_files:
        ts_files = list(TIME_SERIES.glob(f"{ticker.upper()}.csv"))
    if not ts_files:
        ts_files = list(TIME_SERIES.glob(f"*{ticker}*.csv"))
    if not ts_files:
        ts_files = list(TIME_SERIES.glob(f"*{ticker.upper()}*.csv"))
    
    if not ts_files:
        print(f"  No time series file found for {ticker}")
        return None
    
    ts_file = ts_files[0]
    print(f"Loading time series: {ts_file.name}")
    ts_data = pd.read_csv(ts_file)
    
    date_col = None
    for col in ts_data.columns:
        if 'date' in col.lower():
            date_col = col
            ts_data[col] = pd.to_datetime(ts_data[col], utc=True).dt.tz_localize(None).dt.normalize()
            break
    
    if not date_col:
        print(f"  No date column found in time series")
        return None
    
    ts_data = ts_data.sort_values(date_col)
    print(f"  Loaded {len(ts_data)} records")
    
    print("Merging daily with financials...")
    merged_data = merge_daily_with_financials(ts_data, date_col, features_df, ticker)
    
    news_files = list(NEWS_DIR.glob(f"{ticker.upper()}.jsonl"))
    if not news_files:
        news_files = list(NEWS_DIR.glob(f"{ticker}.jsonl"))
    if not news_files:
        news_files = list(NEWS_DIR.glob(f"*{ticker.upper()}*.jsonl"))
    
    if news_files:
        news_file = news_files[0]
        print(f"Loading news: {news_file.name}")
        news_aggregated = load_and_process_news(news_file, ticker)
        
        print("Merging with news...")
        final_merged = merge_with_news(merged_data, news_aggregated, date_col)
    else:
        print("  No news file found")
        final_merged = merged_data
    
    print("Creating windowed samples...")
    final_merged['Date'] = pd.to_datetime(final_merged['Date'])
    final_merged = final_merged.sort_values('Date').reset_index(drop=True)
    
    history_window = config.get('HISTORY_WINDOW_SIZE', 30)
    forecast_horizon = config.get('FORECAST_HORIZON', 5)
    stride = config.get('STRIDE', 1)
    
    samples = create_windowed_samples(
        final_merged, 
        ticker, 
        sector_map,
        history_window=history_window,
        forecast_horizon=forecast_horizon,
        stride=stride
    )
    
    output_file = OUTPUT_PATH / f"{ticker}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {len(samples)} samples at {output_file}")
    return len(samples)


def main():
    """Main execution function"""
    project_root = Path(__file__).parent.parent.resolve()
    config_path = project_root / "config" / "config.yaml"
    
    print(f"Loading config from: {config_path}")
    config = read_yaml(config_path)
    
    GRAPH_DATA = project_root / config['DATA_DICTIONARY']
    sector_df = pd.read_csv(GRAPH_DATA)
    sector_map = dict(zip(sector_df['stock_name'], sector_df['Sector']))
    
    tickers = config.get('TICKERS', [])
    
    if not tickers:
        print("ERROR: No tickers found in config.yaml")
        print("Please add TICKERS list to config.yaml")
        return
    
    print(f"\nProcessing {len(tickers)} ticker(s): {', '.join(tickers)}")
    
    results = {}
    for ticker in tickers:
        try:
            samples_count = process_ticker(ticker, config, project_root, sector_map)
            results[ticker] = samples_count
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            results[ticker] = None
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for ticker, count in results.items():
        status = f"{count} samples" if count else "Failed"
        print(f"  {ticker.upper()}: {status}")


if __name__ == "__main__":
    main()