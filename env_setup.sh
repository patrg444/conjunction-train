#!/usr/bin/env bash
set -e

# Ensure conda is initialized in the script's shell
# Adjust the path below if your conda installation is not in the default location
source ~/anaconda3/etc/profile.d/conda.sh

# Create the conda environment
echo "Creating conda environment 'ted-humor-audio'..."
conda create -y -n ted-humor-audio python=3.9

# Activate the environment
echo "Activating environment..."
conda activate ted-humor-audio

# Install core dependencies (TensorFlow and TensorFlow-IO)
echo "Installing core dependencies (TensorFlow, TensorFlow-IO)..."
pip install tensorflow==2.15.0 tensorflow-io==0.36.0

# Install other dependencies from requirements.txt
echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Environment setup complete."
echo "To activate this environment in a new shell, run: conda activate ted-humor-audio"
