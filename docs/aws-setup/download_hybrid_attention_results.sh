#!/bin/bash
# Download the hybrid Conv1D-TCN model results from EC2
# This script:
# 1. Downloads the model files
# 2. Downloads the training logs
# 3. Creates a local directory structure to organize the results

set -e  # Exit on any error

# EC2 instance details
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_MODEL_PATH="/home/ec2-user/emotion_training/models/hybrid_conv1d_tcn/"
REMOTE_LOG_FILE="/home/ec2-user/emotion_training/hybrid_attention_training.log"
LOCAL_RESULTS_DIR="models/hybrid_conv1d_tcn"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "======================================================"
echo "  DOWNLOADING HYBRID CONV1D-TCN MODEL RESULTS"
echo "======================================================"
echo "Target: $USERNAME@$INSTANCE_IP"
echo "Using key: $KEY_FILE"
echo "Remote model path: $REMOTE_MODEL_PATH"
echo "Remote log file: $REMOTE_LOG_FILE"
echo "Local results directory: $LOCAL_RESULTS_DIR"

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found: $KEY_FILE"
    exit 1
fi

# Create local directories if they don't exist
echo "Creating local directories..."
mkdir -p "$LOCAL_RESULTS_DIR"
mkdir -p "$LOCAL_RESULTS_DIR/logs"

# Check if model files exist on the server
echo "Checking if model files exist on the server..."
MODEL_EXISTS=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "if [ -d $REMOTE_MODEL_PATH ]; then echo 'yes'; else echo 'no'; fi")

if [ "$MODEL_EXISTS" = "no" ]; then
    echo "Warning: Model directory does not exist on the server"
    echo "Check if training has started and created the model directory"
else
    # Download model files
    echo "Downloading model files..."
    scp -i "$KEY_FILE" -r "$USERNAME@$INSTANCE_IP:$REMOTE_MODEL_PATH"* "$LOCAL_RESULTS_DIR/"
    
    # Create model info file
    echo "Creating model info file..."
    cat > "$LOCAL_RESULTS_DIR/model_info.json" << EOF
{
  "model_name": "Hybrid Conv1D-TCN with Cross-Modal Attention",
  "download_date": "$(date)",
  "ec2_instance": "$INSTANCE_IP",
  "architecture": "Conv1D for audio, TCN with self-attention for video, cross-modal attention fusion",
  "datasets": ["RAVDESS", "CREMA-D"],
  "features": ["Audio OpenSMILE", "FaceNet video embeddings"]
}
EOF
fi

# Check if log file exists on the server
echo "Checking if log file exists on the server..."
LOG_EXISTS=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "if [ -f $REMOTE_LOG_FILE ]; then echo 'yes'; else echo 'no'; fi")

if [ "$LOG_EXISTS" = "no" ]; then
    echo "Warning: Log file does not exist on the server"
    echo "Check if training has started and created the log file"
else
    # Download log file
    echo "Downloading log file..."
    scp -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP:$REMOTE_LOG_FILE" "$LOCAL_RESULTS_DIR/logs/training_log_$TIMESTAMP.log"
    
    # Extract validation metrics
    echo "Extracting validation metrics..."
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -a 'val_accuracy' $REMOTE_LOG_FILE" > "$LOCAL_RESULTS_DIR/logs/validation_metrics_$TIMESTAMP.txt"
    
    # Create a summary file
    echo "Creating summary file..."
    echo "Hybrid Conv1D-TCN Model Training Summary" > "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    echo "=========================================" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    echo "Download date: $(date)" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    echo "" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    
    echo "Training Results:" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -a 'Training history summary' -A 5 $REMOTE_LOG_FILE" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    
    echo "" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    echo "Training Time:" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -a 'Training completed in' $REMOTE_LOG_FILE" >> "$LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
fi

echo "======================================================"
echo "  DOWNLOAD COMPLETE"
echo "======================================================"
echo "Results saved to: $LOCAL_RESULTS_DIR"
echo "Log files saved to: $LOCAL_RESULTS_DIR/logs"
echo "Summary file: $LOCAL_RESULTS_DIR/logs/training_summary_$TIMESTAMP.txt"
echo "======================================================"
