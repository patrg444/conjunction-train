#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

cd /home/ec2-user/emotion_training || exit 1 # Change to the training directory
LOG_FILE="training_spectrogram_cnn_lstm_20250329_204315.log"
PID_FILE="spectrogram_cnn_lstm_20250329_204315_pid.txt"

echo "Starting Spectrogram CNN + LSTM model training..." > "\$LOG_FILE"
echo "PID will be written to: \$PID_FILE" >> "\$LOG_FILE"
echo "Script will exit immediately on any command failure." >> "\$LOG_FILE"

# --- Activate Conda ---
echo "Activating Conda environment 'emotion_env'..." >> "\$LOG_FILE"
# Assuming miniconda3 path is correct based on previous attempts
source /home/ec2-user/miniconda3/bin/activate emotion_env >> "\$LOG_FILE" 2>&1
echo "Conda environment activated." >> "\$LOG_FILE"

# --- Run Preprocessing ---
echo "Running preprocessing script (scripts/preprocess_spectrograms.py)..." >> "\$LOG_FILE"
python3 -u scripts/preprocess_spectrograms.py >> "\$LOG_FILE" 2>&1
echo "Preprocessing finished successfully." >> "\$LOG_FILE"

# --- Run Training ---
echo "Launching training script (train_spectrogram_cnn_lstm.py)..." >> "\$LOG_FILE"
# Run in background and detach
nohup python3 -u train_spectrogram_cnn_lstm.py >> "\$LOG_FILE" 2>&1 &
PID=\$!
echo \$PID > "\$PID_FILE"
echo "Training process started with PID: \$PID" >> "\$LOG_FILE"
echo "Logs are being written to: \$LOG_FILE"

# Optionally print Python version
# Run this in a subshell to avoid exiting the main script if it fails
(python3 --version >> "\$LOG_FILE" 2>&1) || echo "Failed to get Python version." >> "\$LOG_FILE"

echo "Launch script finished initiating background training."
