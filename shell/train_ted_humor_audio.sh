#!/bin/bash

# Script to train an audio-only model on the TED-Humor dataset
# using pre-extracted Covarep features

# Ensure the script stops if any command fails
set -e

# Ensure conda is initialized and activate the environment
# Adjust the path below if your conda installation is not in the default location
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ted-humor-audio

echo "Starting TED-Humor audio-only training..."

# Define paths
CONFIG_FILE="configs/train_ted_humor_audio.yaml"
TRAIN_SCRIPT="scripts/train_ted_humor_audio.py" # This script will be created/modified

# 1. Check if the configuration file exists
echo "Checking for configuration file: $CONFIG_FILE"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found. Please ensure '$CONFIG_FILE' exists."
    exit 1
fi

# 2. Check if the training script exists
echo "Checking for training script: $TRAIN_SCRIPT"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found. Please ensure '$TRAIN_SCRIPT' exists."
    echo "You may need to create or modify a training script to work with pre-extracted features."
    exit 1
fi

# 3. Run the training script with the specified configuration
echo "Running training script: $TRAIN_SCRIPT with config $CONFIG_FILE"
python "$TRAIN_SCRIPT" --config "$CONFIG_FILE" --use_cache

echo "Training script finished."
echo "TED-Humor audio-only training pipeline initiated."
echo "Check lightning_logs/ted_humor_audio_only for training logs and checkpoints."
