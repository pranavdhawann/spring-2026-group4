# Spring 2026 – Group 4 Capstone Project

This repository contains the coursework, reports, and implementation for the Spring 2026 Group 4 capstone project.

## 📁 Directory Structure

- `cookbooks/` – Setup guides and usage examples
  - `EDA/` - notebooks related to EDA
- `demo/` – Demo files and experiments
- `presentation/` – Slides and presentation materials
- `reports/` – Project reports and documentation
  - `proposal/` – Capstone project proposal and supporting figures
- `research_paper/` – Research paper drafts and references
- `src/` – Source code for the project


## 🛠️ Setup

### 🐍 Python Environment

1. Navigate to the project root directory:

   ```bash
   cd spring-2026-group4
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:

   ```bash
   source venv/bin/activate   # Linux / macOS
   ```
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### 📦 Dataset Setup

1. Download the dataset from Hugging Face:
   👉 [Multimodal Financial Time-Series Dataset](https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting)
Here’s a polished version of your instructions with corrected grammar, formatting, and clarity:

---

2. Download the Dataset and Setup Environment

You can download the dataset by running the following script:

```bash
chmod +x setup_dataset.sh
./setup_dataset.sh
```

This script will save the dataset in the required format and **skip step 3**.

**Note for Ubuntu users:**
You can run the following commands to download the dataset, set up the virtual environment, and install all required Python packages:

```bash
chmod +x setup_dataset.sh
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

This will automatically:

* Download the dataset
* Set up the Python virtual environment
* Install all necessary Python packages

3. Save the dataset under the `data/` directory at the project root, following this structure:

```text
data/
└── multi-modal-dataset/
    ├── sp500_news/                     # News articles related to S&P 500 stocks
    ├── sp500_time_series/              # Time-series financial data for S&P 500 stocks
    ├── sp500stock_data_description.csv # Metadata and feature descriptions
    └── stock_scores_news_1.csv         # Computed stock scores based on news data
```

4. If your dataset is stored in a different location, update the paths in:

   ```
   config/config.yaml
   ```

5. From the project root directory, run the following command to generate the stock score file:

   ```bash
   python -m setupScripts.cal_data_quality_scores
   ```

   This will generate:

   ```
   data/multi-modal-dataset/stock_scores_news_1.csv
   ```

---

### ⚙️ Configuration Notes

* All dataset paths are centrally managed via `config/config.yaml`.
* Avoid hardcoding paths in scripts; update the config file instead if the dataset location changes.


## 📄 Proposal

The capstone proposal can be found here:
**[View Project Proposal](reports/proposal)**

---

## 👥 Team
Spring 2026 – Group 4 - Pranav Dhawan, Akshit Reddy Palle, Aakash Singh Sivaram, Sayam Palrecha
