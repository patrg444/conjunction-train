#!/bin/bash
# Script to download the LSTM attention model results

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

# ANSI colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

DOWNLOAD_DIR="lstm_attention_model_results"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DOWNLOADING LSTM ATTENTION MODEL RESULTS                    ${NC}"
echo -e "${BLUE}=================================================================${NC}"

# Create download directory
mkdir -p $DOWNLOAD_DIR
echo -e "${YELLOW}Downloading trained model and logs to $DOWNLOAD_DIR...${NC}"

# Download the models and logs
scp -i $KEY_FILE -r ec2-user@$INSTANCE_IP:~/emotion_training/models/attention_focal_loss $DOWNLOAD_DIR/
scp -i $KEY_FILE ec2-user@$INSTANCE_IP:~/emotion_training/$LOG_FILE $DOWNLOAD_DIR/

echo -e "${GREEN}Results downloaded to $DOWNLOAD_DIR directory\033[0m"
echo -e "${BLUE}=================================================================${NC}"
