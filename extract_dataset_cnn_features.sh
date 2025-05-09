#!/bin/bash

# Configurable CNN feature extraction script for any audio emotion dataset
# This script provides a general solution using the fixed preprocessing technique

# Default values
DATASET_NAME="custom"
INPUT_DIR="data/features_spectrogram"
OUTPUT_DIR="data/features_cnn_fixed"
WORKERS=1
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
SERVER="ubuntu@18.208.166.91"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET_NAME="$2"
      shift 2
      ;;
    --input)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --key)
      SSH_KEY="$2"
      shift 2
      ;;
    --server)
      SERVER="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: ./extract_dataset_cnn_features.sh --dataset [name] --input [input_dir] --output [output_dir] --workers [num] --key [ssh_key] --server [user@host]"
      exit 1
      ;;
  esac
done

echo "=== Starting CNN feature extraction for $DATASET_NAME dataset ==="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Workers: $WORKERS"

# SSH to the server and run the feature extraction
ssh -i "$SSH_KEY" "$SERVER" \
  "cd /home/ubuntu/emotion-recognition && \
   python3 fixed_preprocess_cnn_audio_features.py \
   --spectrogram_dir $INPUT_DIR \
   --output_dir $OUTPUT_DIR \
   --workers $WORKERS \
   --verbose"

echo "=== $DATASET_NAME CNN Feature Extraction Completed ==="
echo "Output saved to $OUTPUT_DIR/"

# Add monitoring instructions
echo ""
echo "To monitor progress, use:"
echo "./monitor_ravdess_cnn_extraction.sh"
echo ""
echo "To view extracted features:"
echo "ssh -i $SSH_KEY $SERVER \"ls -la /home/ubuntu/emotion-recognition/$OUTPUT_DIR/ | head\""
