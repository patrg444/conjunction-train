#!/bin/bash

# Simple demo script for running the TensorFlow 2.x compatible emotion recognition model

# Set terminal colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}TensorFlow 2.x Compatible Emotion Recognition Demo${NC}\n"

# Check if model exists
MODEL_PATH="models/dynamic_padding_no_leakage/model_best.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Model file not found at $MODEL_PATH${NC}"
    echo "Please run download_best_model.sh first to download the pre-trained model."
    exit 1
fi

# Run the example script
echo "Running compatible model demo..."
python scripts/load_model_example.py

echo -e "\n${GREEN}To run the full real-time emotion recognition with the compatible model, use:${NC}"
echo "./run_realtime_emotion_compatible.sh"
