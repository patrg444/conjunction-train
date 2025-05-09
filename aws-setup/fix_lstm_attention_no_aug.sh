#!/bin/bash
# Fix script for LSTM attention no augmentation model training
# This will install the missing dependencies and restart the training

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Source connection details
if [ ! -f "aws-setup/lstm_attention_model_connection.txt" ]; then
    echo -e "${RED}Error: Connection details file not found.${NC}"
    echo "Please ensure aws-setup/lstm_attention_model_connection.txt exists."
    exit 1
fi

source aws-setup/lstm_attention_model_connection.txt

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     FIXING LSTM ATTENTION MODEL (NO AUGMENTATION)              ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"

# SSH into the instance and fix the dependencies
echo -e "${YELLOW}Connecting to EC2 instance to fix dependencies...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'ENDSSH'
cd ~/emotion_training

# Kill any existing training processes
echo "Stopping any existing training processes..."
pkill -f train_branched_attention_no_aug.py

# Install the required packages
echo "Installing required Python packages..."
python3 -m pip install --user numpy==1.19.5 tensorflow==2.6.0 scipy==1.7.0 scikit-learn==0.24.2 matplotlib==3.4.2 pandas==1.3.0 urllib3==1.26.6

# Configure environment variables for TensorFlow
echo "export TF_NUM_INTEROP_THREADS=96" >> ~/.bashrc
echo "export TF_NUM_INTRAOP_THREADS=96" >> ~/.bashrc
echo "export OMP_NUM_THREADS=96" >> ~/.bashrc
source ~/.bashrc

# Restart the training
echo "Restarting the training process..."
cd ~/emotion_training
nohup python3 scripts/train_branched_attention_no_aug.py > training_lstm_attention_no_aug.log 2>&1 &

# Check if the training process started
sleep 5
if pgrep -f train_branched_attention_no_aug.py > /dev/null; then
    echo "Training process successfully restarted!"
    PID=$(pgrep -f train_branched_attention_no_aug.py)
    echo "Process ID: $PID"
else
    echo "Failed to restart training process. Check logs for details."
fi

echo "Showing the last 10 lines of the log file:"
tail -n 10 training_lstm_attention_no_aug.log
ENDSSH

echo -e "${GREEN}===============================================================${NC}"
echo -e "${GREEN}Fix script completed. Check if training restarted successfully.${NC}"
echo -e "${GREEN}===============================================================${NC}"
echo -e "${YELLOW}To monitor the training:${NC} ./aws-setup/continuous_training_monitor.sh"
echo -e "${GREEN}===============================================================${NC}"
