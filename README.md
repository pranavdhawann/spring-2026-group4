# Multimodal Techniques for Financial Time-Series Forecasting

GWU Data Science Capstone - Spring 2026  
Pranav Dhawan, Aakash Singh Sivaram, Akshit Reddy Palle, Sayam Palrecha  
Supervised by Dr Amir Jafari

<p align="center">
  <img src="research_paper/figs/main.gif" alt="Project GIF" width ="70%"/>
</p>

## Overview

This repository contains the full experimental pipeline for our capstone project benchmarking multimodal deep learning architectures for short-horizon stock price forecasting on S&P 500 equities.

We systematically compare unimodal temporal encoders, pretrained time-series foundation models, multimodal fusion architectures, and graph neural networks using three complementary data modalities:

- Historical OHLCV price series
- Financial news sentiment
- Structured SEC fundamental features

Our central finding is that the choice of fusion mechanism dominates the choice of temporal encoder. Cross-attention and encoder-decoder fusion between a temporal encoder and FinBERT improve on unimodal baselines, while concatenation-based late fusion underperforms them due to modality imbalance.

<table align="center" width="100%">
  <tr>
    <th width="50%">Plot 1: TSMixer Forecast</th>
    <th width="50%">Plot 2: LSTM Forecast</th>
  </tr>
  <tr>
    <td align="center">
      <img src="\research_paper\figs\forecast_aal_tsmixer.png" alt="forecast_tsmixer" width="100%" />
    </td>
    <td align="center">
      <img src="\research_paper\figs\forecast_aapl_lstm.png" alt="forecast_lstm" width="100%" />
    </td>
  </tr>
</table>

## Repository Structure

```text
spring-2026-group4/
|-- config/             # Model and experiment configuration files (YAML/JSON)
|-- cookbooks/          # Step-by-step notebooks for reproducing key experiments
|-- data/               # Data ingestion, preprocessing, and alignment scripts
|-- demo/               # Interactive demos and inference scripts
|-- experiments/        # Experiment tracking, logs, and ablation results
|-- predict/            # Inference pipelines for all model families
|-- presentation/       # Slide deck and presentation assets
|-- reports/            # Generated evaluation reports and metric summaries
|-- research_paper/     # LaTeX source and compiled capstone report
|-- scripts/            # Utility and automation scripts
|-- src/                # Core source code: models, datasets, trainers, utilities
|-- train/              # Training entrypoints for all model families
|-- requirements.txt    # Python dependencies
|-- .pre-commit-config.yaml
`-- .gitignore
```

## Models Benchmarked

### Unimodal Temporal Baselines (Price Only)

| Model | Description |
|---|---|
| LSTM | 2-layer stacked LSTM with 26 engineered OHLCV features |
| PatchTST | Channel-independent Transformer with patch-based tokenization |
| TSMixer | All-MLP architecture alternating time- and feature-mixing blocks |
| Chronos T5-Large | Pretrained T5 encoder-decoder foundation model (zero-shot) |
| Chronos-2 | Improved tokenization + group attention foundation model |
| TFT | Temporal Fusion Transformer with variable selection and interpretable attention |
| TCN-Transformer | Dilated causal convolutions feeding a 2-layer Transformer encoder |

### Multimodal Fusion (Price + News)

| Model | Fusion Strategy |
|---|---|
| ChronosFinBert | Cross-attention between Chronos-T5 encoder and FinBERT [CLS] embedding |
| TFT-FinBERT | Late concatenation of TFT context + FinBERT news embedding |
| TCN-FinBERT | Late concatenation of TCN context + FinBERT news embedding |
| FinBERT+CNN+Transformer | Encoder-decoder fusion: CNN price encoder + FinBERT -> Transformer decoder |

### Graph Neural Networks (Price + News + Sector Graph)

| Model | Description |
|---|---|
| GATv2 | LSTM node encoder + 2-layer Graph Attention Network v2 with cross-sectional ranking loss |
| EvolveGCN | Dynamic graph convolution evolving parameters over time |

## Dataset

All experiments use the S&P 500 subset of the FinMultiTime dataset (Wenyan 2024), a large-scale multimodal benchmark. We exclude image data and Chinese market (HS300) samples.

| Modality | Size | Records | Frequency |
|---|---:|---:|---|
| Financial Reports (Fundamentals) | 84.04 GB | 2,676 tables | Semi-Annual |
| Stock Price Time Series | 1.83 GB | 4,213 series | Daily |
| Financial News Text | 14.1 GB | 3,351,852 articles | Daily |

Data is not included in this repository. Follow the instructions in `data/README.md` to download and preprocess the dataset.

## Quickstart

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Preprocess data

```bash
python data/preprocess_prices.py --config config/data_config.yaml
python data/preprocess_news.py --config config/data_config.yaml
python data/preprocess_fundamentals.py --config config/data_config.yaml
```

3. Train a model

```bash
# Train a unimodal baseline
python train/train_tft.py --config config/tft_config.yaml

# Train a multimodal model
python train/train_chronos_finbert.py --config config/chronos_finbert_config.yaml

# Train a graph model
python train/train_gatv2.py --config config/gatv2_config.yaml
```

4. Run inference

```bash
python predict/predict.py --model tft --checkpoint <path_to_checkpoint> --ticker AAPL
```

5. Reproduce experiments

See `cookbooks/` for step-by-step notebooks covering each model family and ablation study.

## Key Results (5-Day Forecast Horizon)

| Model | RMSE | MAE |
|---|---:|---:|
| TFT | 0.0673 | 0.0350 |
| TCN-Transformer | 0.0673 | 0.0352 |
| Chronos T5-Large | 1.3669 | 1.2130 |
| Chronos-2 | 1.3505 | 1.1954 |
| LSTM | 1.4043 | 1.0880 |
| ChronosFinBert (60d) | 0.1250 | 0.0420 |
| TCN-FinBERT | 0.8119 | 0.5217 |
| TFT-FinBERT | 0.812 | 0.521 |

Note: Models are evaluated in different output spaces (log-return vs. dollar price). Direct cross-group comparison should be made with care. See the full report in `research_paper/` for a detailed discussion.

## Pre-commit Hooks

This project uses pre-commit for code quality. Install hooks before contributing:

```bash
pip install pre-commit
pre-commit install
```

## Citation

If you use this codebase or the findings from our paper, please cite:

```bibtex
@techreport{dhawan2026multimodal,
  title       = {Multimodal Techniques for Financial Time-Series Forecasting},
  author      = {Dhawan, Pranav and Sivaram, Aakash Singh and Palle, Akshit Reddy and Palrecha, Sayam},
  institution = {The George Washington University, Data Science Program},
  year        = {2026},
  month       = {April},
  type        = {Capstone Report}
}
```

## Acknowledgements

Supervised by Dr Amir Jafari, GWU Data Science Program.  
Dataset: FinMultiTime (Peng et al., 2025) and the S&P 500 Multimodal Financial Dataset (Wenyan, 2024).
