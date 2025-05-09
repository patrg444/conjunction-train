#!/bin/bash
# Script to fix Python dependencies and restart LSTM attention model training

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Existing instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"
LOG_FILE="training_lstm_attention_model.log"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     FIXING DEPENDENCIES AND RESTARTING TRAINING                 ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if instance is available
echo -e "${YELLOW}Checking if the instance is available...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
    echo -e "${RED}Failed to connect to the instance. Please check if it's running.${NC}"
    exit 1
fi
echo -e "${GREEN}Instance is available.${NC}"

# Install required dependencies
echo -e "${YELLOW}Installing required Python packages...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << EOF
    echo "Installing essential Python packages..."
    pip3 install --user numpy tensorflow pandas scikit-learn matplotlib h5py
    
    # Verify installations
    echo "Verifying numpy installation..."
    python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
    
    echo "Verifying tensorflow installation..."
    python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
    
    # Kill any running training processes just in case
    pkill -f train_branched_attention.py || true
    
    # Kill any tmux sessions
    tmux kill-session -t lstm_attention 2>/dev/null || true
    
    # Create model output directory
    mkdir -p ~/emotion_training/models/attention_focal_loss

    # Check for sequence_data_generator.py
    if [ ! -f ~/emotion_training/scripts/sequence_data_generator.py ]; then
        echo "Error: sequence_data_generator.py not found!"
        exit 1
    fi
EOF

# Upload sequence data generator if needed
if ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "[ ! -f ~/emotion_training/scripts/sequence_data_generator.py ]"; then
    echo -e "${YELLOW}Uploading sequence_data_generator.py from local...${NC}"
    if [ -f "scripts/sequence_data_generator.py" ]; then
        scp -i $KEY_FILE scripts/sequence_data_generator.py ec2-user@$INSTANCE_IP:~/emotion_training/scripts/
    else
        echo -e "${RED}Can't find sequence_data_generator.py locally.${NC}"
        exit 1
    fi
fi

# Start training in a tmux session for persistence
echo -e "${YELLOW}Restarting LSTM attention model training...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << EOF
    # Create a shell script to execute training
    cat > ~/emotion_training/run_lstm_attention.sh << 'EOFINNER'
#!/bin/bash
cd ~/emotion_training
echo "Starting LSTM attention model training at \$(date)"
python3 scripts/train_branched_attention.py > $LOG_FILE 2>&1
echo "Training completed with exit code \$? at \$(date)"
EOFINNER

    chmod +x ~/emotion_training/run_lstm_attention.sh

    # Start in tmux
    tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./run_lstm_attention.sh"
    
    echo "LSTM attention model training restarted in tmux session."
    
    # Check if it's running
    sleep 2
    if pgrep -f train_branched_attention.py > /dev/null; then
        echo "Confirmed: Training process is now running."
    else
        echo "Warning: Training process may not have started successfully."
    fi
EOF

echo -e "${GREEN}Dependencies fixed and LSTM attention model training restarted.${NC}"
echo ""
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}                      MONITORING OPTIONS${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}To monitor training:${NC}"
echo -e "bash aws-setup/stream_logs.sh"
echo ""
echo -e "${YELLOW}To check CPU usage:${NC}"
echo -e "bash aws-setup/monitor_cpu_usage.sh"
echo ""
echo -e "${YELLOW}To check training files:${NC}"
echo -e "bash aws-setup/check_training_files.sh"
echo -e "${BLUE}=================================================================${NC}"
