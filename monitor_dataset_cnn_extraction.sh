#!/bin/bash

# Generic CNN feature extraction monitoring script for any dataset
# This script provides monitoring capabilities for the feature extraction process

# Default values
DATASET_NAME="custom"
OUTPUT_DIR="data/features_cnn_fixed"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
SERVER="ubuntu@18.208.166.91"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET_NAME="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
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
      echo "Usage: ./monitor_dataset_cnn_extraction.sh --dataset [name] --output [output_dir] --key [ssh_key] --server [user@host]"
      exit 1
      ;;
  esac
done

echo "=== Monitoring $DATASET_NAME CNN Feature Extraction ==="

# SSH to server and check extraction progress
ssh -i "$SSH_KEY" "$SERVER" \
  "cd /home/ubuntu/emotion-recognition && \
   echo \"Files processed:\" && \
   ls -la $OUTPUT_DIR/ | wc -l && \
   echo 'Progress details:' && \
   ps aux | grep python | grep fixed_preprocess_cnn_audio_features.py"

echo "=== Monitoring Complete ==="
echo "To view files in the output directory:"
echo "ssh -i $SSH_KEY $SERVER \"ls -la /home/ubuntu/emotion-recognition/$OUTPUT_DIR/ | head\""

# Add additional helpful commands
echo ""
echo "To check extraction logs:"
echo "ssh -i $SSH_KEY $SERVER \"cd /home/ubuntu/emotion-recognition && tail -n 50 nohup.out\""
echo ""
echo "To check disk usage:"
echo "ssh -i $SSH_KEY $SERVER \"df -h /home/ubuntu/emotion-recognition/$OUTPUT_DIR\""
