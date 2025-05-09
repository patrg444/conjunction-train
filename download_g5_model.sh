#!/usr/bin/env bash
# Script to download the trained model and logs from G5 instance
# Run this script after training has completed (100 epochs)

# Set constants
SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"
EC2_PROJECT_PATH="/home/$SSH_USER/emotion-recognition"
LOCAL_OUTPUT_DIR="trained_models_$(date +%Y%m%d)"

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== G5 Model Download Script - $(date) =====${NC}"

# Create local directory for downloaded files
mkdir -p "$LOCAL_OUTPUT_DIR/models"
mkdir -p "$LOCAL_OUTPUT_DIR/logs"
mkdir -p "$LOCAL_OUTPUT_DIR/tensorboard"
echo -e "${GREEN}Created local directory: $LOCAL_OUTPUT_DIR${NC}"

# Check SSH connection
echo "Testing SSH connection..."
ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" "echo Connected successfully" || {
    echo -e "${RED}SSH connection failed. Please check your SSH key and security settings.${NC}"
    exit 1
}

# Find the latest model file
echo "Finding latest trained model..."
MODEL_FILE=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/models -name 'audio_pooling_lstm_with_laughter_*.h5' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' '" || echo "")

if [ -z "$MODEL_FILE" ]; then
    echo -e "${RED}No model file found. Training may not have completed or model save format is different.${NC}"
    echo -e "${YELLOW}Checking alternative model file patterns...${NC}"
    MODEL_FILE=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/models -name '*.h5' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' '" || echo "")
    
    if [ -z "$MODEL_FILE" ]; then
        echo -e "${RED}Could not find any .h5 model files.${NC}"
    else
        echo -e "${GREEN}Found alternate model file: $MODEL_FILE${NC}"
    fi
else
    echo -e "${GREEN}Found model file: $MODEL_FILE${NC}"
fi

# If model found, download it
if [ -n "$MODEL_FILE" ]; then
    echo "Downloading model file..."
    scp -i "$SSH_KEY" "$SSH_HOST:$MODEL_FILE" "$LOCAL_OUTPUT_DIR/models/"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model downloaded successfully to $LOCAL_OUTPUT_DIR/models/$(basename "$MODEL_FILE")${NC}"
    else
        echo -e "${RED}Failed to download model file.${NC}"
    fi
fi

# Find and download the training logs
echo "Finding training logs..."
ssh -i "$SSH_KEY" "$SSH_HOST" "mkdir -p $EC2_PROJECT_PATH/logs" # Ensure logs directory exists
LOG_FILES=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/logs -name 'train_laugh_*.log' -o -name 'training_*.log' -type f" || echo "")

if [ -z "$LOG_FILES" ]; then
    echo -e "${RED}No training log files found.${NC}"
else
    echo "Downloading training logs..."
    for LOG_FILE in $LOG_FILES; do
        echo "Downloading $LOG_FILE..."
        scp -i "$SSH_KEY" "$SSH_HOST:$LOG_FILE" "$LOCAL_OUTPUT_DIR/logs/"
    done
    echo -e "${GREEN}Training logs downloaded to $LOCAL_OUTPUT_DIR/logs/${NC}"
fi

# Download TensorBoard logs if available
echo "Checking for TensorBoard logs..."
TENSORBOARD_EXISTS=$(ssh -i "$SSH_KEY" "$SSH_HOST" "if [ -d $EC2_PROJECT_PATH/logs/tensorboard ]; then echo 'exists'; else echo ''; fi" || echo "")

if [ -n "$TENSORBOARD_EXISTS" ]; then
    echo "Downloading TensorBoard logs..."
    scp -i "$SSH_KEY" -r "$SSH_HOST:$EC2_PROJECT_PATH/logs/tensorboard" "$LOCAL_OUTPUT_DIR/tensorboard/"
    echo -e "${GREEN}TensorBoard logs downloaded to $LOCAL_OUTPUT_DIR/tensorboard/${NC}"
else
    echo -e "${YELLOW}No TensorBoard logs directory found.${NC}"
fi

# Download normalization stats if available
echo "Checking for normalization statistics..."
NORM_FILES=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/models -name '*normalization_stats.pkl'" || echo "")

if [ -n "$NORM_FILES" ]; then
    echo "Downloading normalization statistics..."
    mkdir -p "$LOCAL_OUTPUT_DIR/normalization"
    for NORM_FILE in $NORM_FILES; do
        echo "Downloading $NORM_FILE..."
        scp -i "$SSH_KEY" "$SSH_HOST:$NORM_FILE" "$LOCAL_OUTPUT_DIR/normalization/"
    done
    echo -e "${GREEN}Normalization statistics downloaded to $LOCAL_OUTPUT_DIR/normalization/${NC}"
else
    echo -e "${YELLOW}No normalization statistics files found.${NC}"
fi

# Create summary file with information
echo "Creating summary file..."
SUMMARY_FILE="$LOCAL_OUTPUT_DIR/model_summary.txt"

{
    echo "===== G5 Model Training Summary ====="
    echo "Download date: $(date)"
    echo "EC2 instance: $AWS_IP"
    echo ""
    echo "=== Model ==="
    if [ -n "$MODEL_FILE" ]; then
        echo "Model file: $(basename "$MODEL_FILE")"
        echo "Model size: $(du -h "$LOCAL_OUTPUT_DIR/models/"* | cut -f1)"
    else
        echo "No model file found."
    fi
    echo ""
    echo "=== Logs ==="
    if [ -n "$LOG_FILES" ]; then
        echo "Log files: $(ls -1 "$LOCAL_OUTPUT_DIR/logs/" | wc -l)"
        LATEST_LOG=$(ls -t "$LOCAL_OUTPUT_DIR/logs/"* 2>/dev/null | head -1 || echo "")
        if [ -n "$LATEST_LOG" ]; then
            echo "Latest log: $(basename "$LATEST_LOG")"
            echo "Training epochs completed: $(grep -o 'Epoch [0-9]*/100' "$LATEST_LOG" | tail -1 || echo "Unknown")"
            echo "Final accuracy: $(grep -o 'val_emotion_output_accuracy: [0-9.]*' "$LATEST_LOG" | tail -1 || echo "Unknown")"
            echo "Final laughter accuracy: $(grep -o 'val_laugh_output_accuracy: [0-9.]*' "$LATEST_LOG" | tail -1 || echo "Unknown")"
        fi
    else
        echo "No log files found."
    fi
    echo ""
    echo "=== Other ====="
    if [ -n "$TENSORBOARD_EXISTS" ]; then
        echo "TensorBoard logs downloaded."
    else
        echo "No TensorBoard logs found."
    fi
    if [ -n "$NORM_FILES" ]; then
        echo "Normalization statistics downloaded."
    else
        echo "No normalization statistics found."
    fi
} > "$SUMMARY_FILE"

echo -e "${GREEN}Summary file created at $SUMMARY_FILE${NC}"
echo -e "${GREEN}Download complete! All files saved to $LOCAL_OUTPUT_DIR${NC}"
