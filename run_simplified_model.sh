#!/bin/bash

# Run simplified CNN-LSTM model with feature normalization
# Uses a reduced number of epochs for quick testing

# Make directory for model checkpoints if it doesn't exist
mkdir -p models

# Run with 2 epochs instead of 100 for testing purposes
EPOCHS=2 \
BATCH_SIZE=32 \
PATIENCE=2 \
python simplified_cnn_lstm_model.py
