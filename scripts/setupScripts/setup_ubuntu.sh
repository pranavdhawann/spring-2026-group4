#!/bin/bash
# setup_ubuntu.sh
# This script sets up the Python environment, installs dependencies, and runs the dataset setup.

set -e  # Exit immediately if a command exits with a non-zero status

# Update package lists and install python3-venv
echo "Updating package lists..."
sudo apt update

echo "Installing python3.12-venv and unzip..."
sudo apt install -y python3.12-venv unzip

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping pip install."
fi

# Run dataset setup script
if [ -f "setup_dataset.sh" ]; then
    echo "Running dataset setup..."
    chmod +x setup_dataset.sh
    ./setup_dataset.sh
else
    echo "setup_dataset.sh not found, skipping dataset setup."
fi

echo "Setup complete!"