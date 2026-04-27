#!/bin/bash

set -e  # Exit immediately if a command fails

BASE_DIR="data/multi-modal-dataset"

echo "Creating base directory..."
mkdir -p "$BASE_DIR"

############################################
# Function to download, unzip, and flatten
############################################
download_and_flatten () {
    NAME=$1
    URL=$2

    echo "----------------------------------------"
    echo "Processing $NAME ..."
    echo "----------------------------------------"

    TARGET_DIR="$BASE_DIR/$NAME"
    TMP_DIR="$BASE_DIR/${NAME}_tmp"
    ZIP_FILE="$BASE_DIR/${NAME}.zip"

    mkdir -p "$TARGET_DIR"

    echo "Downloading $NAME ..."
    wget -c --show-progress -O "$ZIP_FILE" "$URL"

    echo "Extracting $NAME ..."
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"
    unzip -q "$ZIP_FILE" -d "$TMP_DIR"

    # Find inner directory (if exists) and flatten
    INNER_DIR=$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -d "$INNER_DIR" ]; then
        mv "$INNER_DIR"/* "$TARGET_DIR"/
    else
        mv "$TMP_DIR"/* "$TARGET_DIR"/
    fi

    rm -rf "$TMP_DIR"
    rm -f "$ZIP_FILE"

    echo "✅ $NAME completed"
    echo ""
}

############################################
# 1️⃣ Time Series
############################################
download_and_flatten \
"sp500_time_series" \
"https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting/resolve/main/time_series/S%26P500_time_series.zip?download=true"

############################################
# 2️⃣ News
############################################
download_and_flatten \
"sp500_news" \
"https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting/resolve/main/text/sp500_news.zip?download=true"

############################################
# 3️⃣ Tabular
############################################
download_and_flatten \
"sp500_table" \
"https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting/resolve/main/table/SP500_tabular.zip?download=true"

############################################
# 4️⃣ CSV Description (no unzip needed)
############################################
echo "Downloading dataset description CSV..."
wget -c --show-progress -O "$BASE_DIR/sp500stock_data_description.csv" \
"https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting/resolve/main/sp500stock_data_description.csv?download=true"

echo ""
echo "🎉 All files downloaded, flattened, and organized successfully!"
