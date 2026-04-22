#!/bin/bash

# Notify the user
echo "⚠️  Please make sure to run this script from the project root directory."

# Run the Python scripts
echo "Running scripts to calculate data quality scores..."
python3 -m setupScripts.cal_data_quality_scores

echo "Creating baseline dataset..."
python3 -m setupScripts.baseline_dataset_creation

echo "✅ Scripts executed successfully."
