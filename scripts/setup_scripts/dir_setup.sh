#!/usr/bin/env bash
# =============================================================================
# setup.sh
# Downloads the FinMultiTime multimodal financial dataset from HuggingFace,
# skipping images, then:
#   1. Downloads everything except images
#   2. Unzips S&P 500 archives only
#   3. Deletes all zip files
#   4. Renames output dir to 'data'
#   5. Removes hs300stock_data_description.csv
#   6. Flattens time_series/ and text/ subdirs, renames them
#   7. Lifts financial report subdirs out of table/financial_reports/
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATASET_ID="Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting"
DOWNLOAD_DIR="${1:-./dataset}"   # temporary download dir
DATA_DIR="$(dirname "${DOWNLOAD_DIR}")/data"  # final dir name

# Colours
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 0. Pre-flight checks
# --------------------------------------------------------------------------- #
log "Checking dependencies..."

command -v python3 &>/dev/null || die "python3 not found. Please install Python 3.8+."

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    warn "huggingface_hub not found — installing into current environment..."
    pip install --quiet huggingface_hub || die "Failed to install huggingface_hub. Activate your venv first."
fi

command -v unzip &>/dev/null || die "unzip not found. Run: sudo apt install -y unzip"

ok "All dependencies satisfied."

# --------------------------------------------------------------------------- #
# 1. Download dataset — everything except images
# --------------------------------------------------------------------------- #
log "Downloading dataset to '${DOWNLOAD_DIR}' (excluding images)..."

python3 - <<PYEOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${DATASET_ID}",
    repo_type="dataset",
    local_dir="${DOWNLOAD_DIR}",
    ignore_patterns=[
        "image/**",
        "*.jpg", "*.jpeg", "*.png",
        "*.gif", "*.webp", "*.bmp", "*.tiff",
    ],
)
PYEOF

ok "Download complete."

# --------------------------------------------------------------------------- #
# 2. Unzip S&P 500 archives only
# --------------------------------------------------------------------------- #
log "Unzipping S&P 500 archives..."

unzip_if_exists() {
    local zip_path="$1"
    local out_dir="$2"
    if [ -f "${zip_path}" ]; then
        log "  Extracting: ${zip_path}"
        unzip -q "${zip_path}" -d "${out_dir}"
        ok "  Done: ${zip_path}"
    else
        warn "  Not found (skipping): ${zip_path}"
    fi
}

unzip_if_exists "${DOWNLOAD_DIR}/table/SP500_tabular.zip"            "${DOWNLOAD_DIR}/table/"
unzip_if_exists "${DOWNLOAD_DIR}/text/sp500_news.zip"                "${DOWNLOAD_DIR}/text/"
unzip_if_exists "${DOWNLOAD_DIR}/time_series/S&P500_time_series.zip" "${DOWNLOAD_DIR}/time_series/"

ok "S&P 500 archives extracted."

# --------------------------------------------------------------------------- #
# 3. Delete all zip files
# --------------------------------------------------------------------------- #
log "Removing zip files..."
ZIP_COUNT=$(find "${DOWNLOAD_DIR}" -name "*.zip" | wc -l)
if [ "${ZIP_COUNT}" -gt 0 ]; then
    find "${DOWNLOAD_DIR}" -name "*.zip" -delete
    ok "Deleted ${ZIP_COUNT} zip file(s)."
else
    warn "No zip files found."
fi

# --------------------------------------------------------------------------- #
# 4. Rename download dir → data/
# --------------------------------------------------------------------------- #
log "Renaming '${DOWNLOAD_DIR}' → '${DATA_DIR}'..."
mv "${DOWNLOAD_DIR}" "${DATA_DIR}"
ok "Renamed to '${DATA_DIR}'."

# --------------------------------------------------------------------------- #
# 5. Remove HS300 CSV
# --------------------------------------------------------------------------- #
log "Removing hs300stock_data_description.csv..."
if [ -f "${DATA_DIR}/hs300stock_data_description.csv" ]; then
    rm "${DATA_DIR}/hs300stock_data_description.csv"
    ok "Removed hs300stock_data_description.csv."
else
    warn "hs300stock_data_description.csv not found (skipping)."
fi

# --------------------------------------------------------------------------- #
# 6. Flatten time_series/ → sp500_timeseries/
# --------------------------------------------------------------------------- #
log "Flattening time_series/ subdirs..."
find "${DATA_DIR}/time_series" -mindepth 2 -type f -exec mv {} "${DATA_DIR}/time_series/" \;
find "${DATA_DIR}/time_series" -mindepth 1 -type d -delete
log "Renaming time_series/ → sp500_timeseries/..."
mv "${DATA_DIR}/time_series" "${DATA_DIR}/sp500_timeseries"
ok "Done: sp500_timeseries/"

# --------------------------------------------------------------------------- #
# 7. Flatten text/ (news) → sp500_news/
# --------------------------------------------------------------------------- #
log "Flattening text/ subdirs..."
find "${DATA_DIR}/text" -mindepth 2 -type f -exec mv {} "${DATA_DIR}/text/" \;
find "${DATA_DIR}/text" -mindepth 1 -type d -delete
log "Renaming text/ → sp500_news/..."
mv "${DATA_DIR}/text" "${DATA_DIR}/sp500_news"
ok "Done: sp500_news/"

# --------------------------------------------------------------------------- #
# 8. Lift financial report subdirs out of table/financial_reports/
# --------------------------------------------------------------------------- #
log "Restructuring table/financial_reports/..."
FINREP="${DATA_DIR}/table/financial_reports"
if [ -d "${FINREP}" ]; then
    find "${FINREP}" -mindepth 1 -maxdepth 1 -exec mv {} "${DATA_DIR}/table/" \;
    rmdir "${FINREP}"
    ok "financial_reports/ contents moved to table/ and wrapper removed."
else
    warn "table/financial_reports/ not found (skipping)."
fi

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
echo ""
ok "================================"
ok " Setup complete!"
ok " Data location: ${DATA_DIR}"
ok "================================"
echo ""
log "Final directory structure:"
find "${DATA_DIR}" -maxdepth 2 | sort
