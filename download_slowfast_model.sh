#!/bin/bash
# Download trained SlowFast model from EC2
# This script downloads the best trained model from the EC2 instance

# Configuration
KEY=~/Downloads/gpu-key.pem
EC2_HOST="ubuntu@54.162.134.77"
REMOTE_DIR="/home/ubuntu/emotion_slowfast"
LOCAL_DIR="./saved_models/slowfast"

# Timestamp for organized downloads
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Echo with color and timestamp
function log() {
    local GREEN='\033[0;32m'
    local NC='\033[0m' # No Color
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Create local directory if it doesn't exist
mkdir -p $LOCAL_DIR

# Check if remote directory exists
log "Checking if remote model directory exists..."
ssh -i $KEY $EC2_HOST "ls -la $REMOTE_DIR" > /dev/null
if [ $? -ne 0 ]; then
    log "Error: Remote directory $REMOTE_DIR not found!"
    exit 1
fi

# List available model files
log "Available model files on EC2:"
ssh -i $KEY $EC2_HOST "find $REMOTE_DIR -name '*.pt' | sort"

# By default, download the best model
log "Downloading best model..."
mkdir -p "$LOCAL_DIR/slowfast_$TIMESTAMP"
scp -i $KEY $EC2_HOST:"$REMOTE_DIR/*_best.pt" "$LOCAL_DIR/slowfast_$TIMESTAMP/"

# Download model config and history
log "Downloading model config and history..."
scp -i $KEY $EC2_HOST:"$REMOTE_DIR/*.yaml" "$LOCAL_DIR/slowfast_$TIMESTAMP/" 2>/dev/null
scp -i $KEY $EC2_HOST:"$REMOTE_DIR/*.json" "$LOCAL_DIR/slowfast_$TIMESTAMP/" 2>/dev/null

# Check if download was successful
if [ -z "$(ls -A $LOCAL_DIR/slowfast_$TIMESTAMP/)" ]; then
    log "No model files found or download failed!"
    log "Trying to find any PT files..."
    ssh -i $KEY $EC2_HOST "find $REMOTE_DIR -name '*.pt' -type f -print0 | xargs -0 ls -lt | head -3"
    log "You can manually specify a model to download with:"
    echo "scp -i $KEY $EC2_HOST:\"$REMOTE_DIR/specific_model_name.pt\" \"$LOCAL_DIR/\""
    rmdir "$LOCAL_DIR/slowfast_$TIMESTAMP"
    exit 1
fi

log "Model files downloaded successfully to $LOCAL_DIR/slowfast_$TIMESTAMP/"
ls -la "$LOCAL_DIR/slowfast_$TIMESTAMP/"

# Optional: download validation metrics if available
if ssh -i $KEY $EC2_HOST "ls $REMOTE_DIR/*_accuracy.json 2>/dev/null"; then
    log "Downloading validation metrics..."
    scp -i $KEY $EC2_HOST:"$REMOTE_DIR/*_accuracy.json" "$LOCAL_DIR/slowfast_$TIMESTAMP/"
fi

log "Download complete!"
