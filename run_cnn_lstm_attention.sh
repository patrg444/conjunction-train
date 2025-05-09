#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Make the script executable
chmod +x cnn_lstm_attention_model.py

# Run the improved CNN-LSTM model with attention mechanism
echo "Starting improved CNN-LSTM model with attention mechanism..."
python cnn_lstm_attention_model.py

echo "Training completed. Check the models directory for results."
