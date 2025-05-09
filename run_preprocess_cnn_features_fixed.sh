#!/bin/bash
# Script to run CNN audio feature extraction in CPU-only mode
# This script avoids CUDA initialization errors by forcing CPU mode

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}======================================================${NC}"
echo -e "${YELLOW}     CNN AUDIO FEATURE EXTRACTION - CPU MODE FIX      ${NC}"
echo -e "${YELLOW}======================================================${NC}"

# Check if already running
PROCESS_COUNT=$(ps aux | grep 'preprocess_cnn_audio_features.py' | grep -v grep | wc -l)
if [ $PROCESS_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Warning: CNN feature extraction is already running!${NC}"
    echo "Process details:"
    ps aux | grep 'preprocess_cnn_audio_features.py' | grep -v grep
    
    echo ""
    echo -e "${YELLOW}Do you want to continue anyway? This will start an additional extraction process.${NC}"
    read -p "Continue? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation canceled."
        exit 1
    fi
fi

# Make sure the output directories exist
mkdir -p ~/emotion-recognition/data/ravdess_features_cnn_audio/
mkdir -p ~/emotion-recognition/data/crema_d_features_cnn_audio/

# Create log directory if it doesn't exist
mkdir -p ~/emotion-recognition/logs/

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=~/emotion-recognition/logs/preprocess_cnn_audio_${TIMESTAMP}.log

echo -e "${GREEN}Starting CNN audio feature extraction in CPU-only mode...${NC}"
echo "Log file: $LOG_FILE"
echo ""
echo -e "${YELLOW}This process will run in the background and may take 1-2 hours${NC}"
echo "To monitor progress, use: ./monitor_cnn_feature_extraction.sh"
echo ""

# Launch extraction in the background with CPU-only mode
cd ~/emotion-recognition
CUDA_VISIBLE_DEVICES=-1 python3 -u scripts/preprocess_cnn_audio_features.py --verbose > $LOG_FILE 2>&1 &

# Wait a moment to check if the process started
sleep 3

# Verify the process started successfully
if ps aux | grep -v grep | grep 'preprocess_cnn_audio_features.py' > /dev/null; then
    PID=$(pgrep -f 'preprocess_cnn_audio_features.py' | head -1)
    echo -e "${GREEN}✅ Feature extraction started successfully with PID: $PID${NC}"
    echo -e "Run ${YELLOW}./monitor_cnn_feature_extraction.sh${NC} to monitor progress"
else
    echo -e "${RED}❌ Failed to start feature extraction process.${NC}"
    echo "Check the log file for errors: $LOG_FILE"
    tail -20 $LOG_FILE
fi
