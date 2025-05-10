#!/bin/bash

# Script to run the CNN-LSTM model with balanced datasets
# Created by Cline to address the data imbalance issue

echo "Starting CNN-LSTM model with balanced dataset..."
echo "RAVDESS: $(find data/ravdess_features_cnn_fixed -type f | wc -l) files"
echo "CREMA-D: $(find data/crema_d_features_cnn_fixed -type f | wc -l) files"

# Run the fixed spectrogram CNN pooling LSTM model
# The model is already pointing to the correct directories: 
# - data/ravdess_features_cnn_fixed
# - data/crema_d_features_cnn_fixed
python scripts/train_spectrogram_cnn_pooling_lstm.py
