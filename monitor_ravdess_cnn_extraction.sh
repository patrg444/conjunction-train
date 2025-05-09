#!/bin/bash

# Script to monitor CNN feature extraction progress for RAVDESS dataset

echo "=== Monitoring RAVDESS CNN Feature Extraction ==="

# SSH to server and check extraction progress
ssh -i /Users/patrickgloria/Downloads/gpu-key.pem ubuntu@18.208.166.91 \
  "cd /home/ubuntu/emotion-recognition && \
   ls -la data/ravdess_features_cnn_fixed/ | wc -l && \
   echo 'Progress details:' && \
   ps aux | grep python | grep fixed_preprocess_cnn_audio_features.py"

echo "=== Monitoring Complete ==="
echo "To view files in the output directory:"
echo "ssh -i /Users/patrickgloria/Downloads/gpu-key.pem ubuntu@18.208.166.91 \"ls -la /home/ubuntu/emotion-recognition/data/ravdess_features_cnn_fixed/ | head\""
