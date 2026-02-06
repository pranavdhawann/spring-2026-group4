# EDA - Exploratory Data Analysis

This directory contains exploratory data analysis scripts and notebooks for S&P500 stock analysis.

## Files

### `eda_utils.py`
Utility functions for one-time analysis: quality scoring, ticker list management, and data loading.

### `yoy_analysis.py`
Calculates year-over-year metrics (returns, volatility, completeness) for all stocks from 2000-2025. Run this first.

### `stock_ranking.py`
Ranks all stocks by data quality score based on completeness and historical coverage (2009-2025).

### `stock_eda_1.ipynb`
Jupyter notebook for interactive data exploration and visualizations.

## Usage

```bash
cd cookbooks/EDA
python3 yoy_analysis.py
python3 stock_ranking.py
```
