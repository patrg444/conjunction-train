#!/bin/bash
# Deploy final fixed CNN-LSTM model with fixed sequence length
# This script transfers our working model to EC2 and starts training

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# EC2 connection details
EC2_KEY="$HOME/Downloads/gpu-key.pem"
EC2_INSTANCE="ubuntu@54.162.134.77"
REMOTE_DIR="/home/ubuntu/emotion_project"

# Local files to transfer
MODEL_FILE="final_cnn_lstm_model.py"
TEST_FILE="test_cnn_lstm_locally.py"  # Include the test script as well

echo -e "${BLUE}Deploying final CNN-LSTM model with fixed sequence length${NC}"
echo -e "${YELLOW}This model fixes the previous issues by:${NC}"
echo -e "  - Using a fixed sequence length (MAX_SEQ_LENGTH=20) for all batches"
echo -e "  - Ensuring proper data padding/truncation in the data generator"
echo -e "  - Using GlobalAveragePooling1D to collapse the time dimension"
echo -e "  - Including enhanced regularization to combat overfitting"

# Check if the key file exists
if [ ! -f "$EC2_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at $EC2_KEY${NC}"
    exit 1
fi

# Check if model files exist
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_FILE${NC}"
    exit 1
fi

# Test SSH connection
echo -e "${BLUE}Testing connection to EC2 instance...${NC}"
ssh -i "$EC2_KEY" -o ConnectTimeout=10 "$EC2_INSTANCE" "echo Connection successful!" || {
    echo -e "${RED}Error: Could not connect to EC2 instance${NC}"
    exit 1
}

# Create remote directory if it doesn't exist
echo -e "${BLUE}Creating remote directory if needed...${NC}"
ssh -i "$EC2_KEY" "$EC2_INSTANCE" "mkdir -p $REMOTE_DIR"

# Transfer model file to EC2
echo -e "${BLUE}Transferring model files to EC2...${NC}"
scp -i "$EC2_KEY" "$MODEL_FILE" "$EC2_INSTANCE:$REMOTE_DIR/final_cnn_lstm_model.py"
scp -i "$EC2_KEY" "$TEST_FILE" "$EC2_INSTANCE:$REMOTE_DIR/test_cnn_lstm_locally.py"

# Make the scripts executable on the remote server
echo -e "${BLUE}Setting execution permissions...${NC}"
ssh -i "$EC2_KEY" "$EC2_INSTANCE" "chmod +x $REMOTE_DIR/final_cnn_lstm_model.py $REMOTE_DIR/test_cnn_lstm_locally.py"

# Create data symlinks if needed
echo -e "${BLUE}Setting up data symlinks if needed...${NC}"
ssh -i "$EC2_KEY" "$EC2_INSTANCE" "
    # Create data directory
    mkdir -p $REMOTE_DIR/data
    
    # Create symlinks to actual data locations if they don't exist
    if [ ! -L $REMOTE_DIR/data/ravdess_features_cnn_fixed ]; then
        ln -s /home/ubuntu/emotion-recognition/data/ravdess_features_cnn_fixed $REMOTE_DIR/data/ravdess_features_cnn_fixed
    fi
    
    if [ ! -L $REMOTE_DIR/data/crema_d_features_cnn_fixed ]; then
        ln -s /home/ubuntu/emotion-recognition/data/crema_d_features_cnn_fixed $REMOTE_DIR/data/crema_d_features_cnn_fixed
    fi
    
    # Create models directory
    mkdir -p $REMOTE_DIR/models
"

# Run test script to verify the model works on EC2
echo -e "${BLUE}Running test script on EC2 to verify model works...${NC}"
ssh -i "$EC2_KEY" "$EC2_INSTANCE" "cd $REMOTE_DIR && python3 test_cnn_lstm_locally.py" || {
    echo -e "${RED}Error: Test script failed on EC2. Aborting training launch.${NC}"
    exit 1
}

# Start the training in a screen session
echo -e "${BLUE}Starting fixed CNN-LSTM training on EC2...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="final_cnn_lstm_training_${TIMESTAMP}.log"

ssh -i "$EC2_KEY" "$EC2_INSTANCE" "
    cd $REMOTE_DIR
    screen -dmS final_cnn_lstm bash -c \"
        echo 'Starting training at $(date)' > $LOG_FILE
        python3 final_cnn_lstm_model.py 2>&1 | tee -a $LOG_FILE
        echo 'Training completed at $(date)' >> $LOG_FILE
    \"
"

echo -e "${GREEN}Final CNN-LSTM model deployment complete!${NC}"
echo -e "${YELLOW}Training is running in a screen session on the EC2 instance.${NC}"
echo -e "Monitor training progress with: ${BLUE}ssh -i \"$HOME/Downloads/gpu-key.pem\" $EC2_INSTANCE \"tail -f $REMOTE_DIR/$LOG_FILE\"${NC}"
echo 
echo -e "Log file: ${BLUE}$REMOTE_DIR/$LOG_FILE${NC}"
