#!/bin/bash
# Monitor the simplified CNN-LSTM training progress
ssh -i "/Users/patrickgloria/Downloads/gpu-key.pem" ubuntu@54.162.134.77 "tail -f /home/ubuntu/emotion_project/simplified_cnn_lstm_training_20250423_115239.log"
