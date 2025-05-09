#!/bin/bash

# Script to train a WavLM-Large audio-only model on the UR-FUNNY dataset

# Ensure the script stops if any command fails
set -e

echo "Starting UR-FUNNY audio-only training with WavLM-Large..."

# Define paths
UR_FUNNY_JSON="datasets/humor_datasets/ur_funny/ur_funny_final.json"
RAW_AUDIO_DIR="datasets/ur_funny/raw_audio"
SEGMENTED_AUDIO_DIR="datasets/ur_funny/audio_segments"
MANIFEST_OUTPUT_DIR="datasets/manifests/humor"
TRAIN_MANIFEST="$MANIFEST_OUTPUT_DIR/ur_funny_train_audio.csv"
VAL_MANIFEST="$MANIFEST_OUTPUT_DIR/ur_funny_val_audio.csv"
CONFIG_FILE="configs/train_ur_funny_wavlm.yaml"
TRAIN_SCRIPT="scripts/train_wav2vec_audio_only_fixed_v4.py"

# 1. Ensure UR-FUNNY JSON is downloaded (if not already)
echo "Checking for UR-FUNNY JSON..."
if [ ! -f "$UR_FUNNY_JSON" ]; then
    echo "UR-FUNNY JSON not found. Running download script..."
    # The download_ur_funny.py script is located at the project root
    python download_ur_funny.py
else
    echo "UR-FUNNY JSON found."
fi

# 2. Fetch raw audio files from YouTube links
echo "Fetching raw audio files..."
bash datasets/scripts/fetch_ur_funny_audio.sh

# 3. Build the manifest and segment audio clips
echo "Building manifest and segmenting audio..."
python datasets/scripts/build_ur_funny_manifest.py

# Check if manifests were created
if [ ! -f "$TRAIN_MANIFEST" ] || [ ! -f "$VAL_MANIFEST" ]; then
    echo "Error: Manifest files were not created. Aborting training."
    exit 1
fi

# 4. Run the training script
echo "Running training script: $TRAIN_SCRIPT with config $CONFIG_FILE"
python "$TRAIN_SCRIPT" --config "$CONFIG_FILE"

echo "Training script finished."
echo "UR-FUNNY audio-only training pipeline completed."
echo "Check lightning_logs/ur_funny_wavlm_audio_only for training logs and checkpoints."
