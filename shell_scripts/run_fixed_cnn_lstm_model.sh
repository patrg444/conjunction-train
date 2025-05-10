#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Make the script executable
chmod +x fixed_cnn_lstm_model.py

# Run the fixed CNN-LSTM model
echo "Starting improved CNN-LSTM model with variable sequence handling..."
python fixed_cnn_lstm_model.py

echo "Training completed. Check the models directory for results."
