#!/bin/bash
# Monitoring script for Branched LSTM/Conv1D model training
# Generated specifically for run started around: 20250403_114037

# ANSI colors
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m' # No Color

INSTANCE_IP="18.206.48.166"
USERNAME="ec2-user"
KEY_FILE="/Users/patrickgloria/conjunction-train/aws-setup/emotion-recognition-key-fixed-20250323090016.pem" # Use absolute path
REMOTE_DIR="/home/ec2-user/emotion_training"
REMOTE_MONITOR_HELPER="monitor_helper_branched_lstm_conv1d_20250403_114037.sh" # Name of the NEW helper script on EC2

echo -e "\${BLUE}==================================================================\${NC}"
echo -e "\${GREEN}    MONITORING BRANCHED LSTM/Conv1D MODEL TRAINING (20250403_114037)    \${NC}"
echo -e "\${BLUE}==================================================================\${NC}"
echo -e "\${YELLOW}Instance:\${NC} \$USERNAME@\$INSTANCE_IP"
echo -e "\${YELLOW}Executing remote helper:\${NC} \$REMOTE_DIR/\$REMOTE_MONITOR_HELPER"
echo -e "\${BLUE}==================================================================\${NC}"

# Start continuous monitoring by executing the remote helper script
echo -e "\${YELLOW}Starting continuous real-time monitoring... Press Ctrl+C to exit.\${NC}"
ssh -t -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "$REMOTE_DIR/$REMOTE_MONITOR_HELPER"
