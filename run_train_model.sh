#!/bin/bash

# Run a short training with reduced epochs to verify model works with local CNN features
# Sets environment variables to override defaults in the script

# Make directory for model checkpoints if it doesn't exist
mkdir -p models

# Run with 2 epochs instead of 150 for testing purposes
EPOCHS=2 \
BATCH_SIZE=24 \
PATIENCE=2 \
python scripts/train_spectrogram_cnn_pooling_lstm.py
