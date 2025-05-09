#!/bin/bash

# Script to run the fixed CNN-LSTM model locally

echo "Starting CNN-LSTM model training with fixed code..."
echo "Using data from data/ravdess_features_cnn_fixed and data/crema_d_features_cnn_fixed"

# Create models directory if it doesn't exist
mkdir -p models

# Run the script
python scripts/train_spectrogram_cnn_pooling_lstm.py

echo "Training complete!"
