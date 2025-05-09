#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get EC2 instance IP
EC2_IP=$(cat aws_instance_ip.txt)

# SSH key path
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo -e "${BLUE}Deploying final fixed XLM-RoBERTa v2 training script to EC2 instance...${NC}"

# Upload the fixed Python script to EC2
echo -e "${BLUE}Uploading v2 fixed script to EC2...${NC}"
scp -i "$SSH_KEY" fixed_train_xlm_roberta_script_v2.py ubuntu@$EC2_IP:/home/ubuntu/train_xlm_roberta_large_extended.py

# Check if the training process is running and kill it if necessary
echo -e "${BLUE}Checking for running training process...${NC}"
RUNNING_PID=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "ps aux | grep python | grep train_xlm_roberta_large_extended.py | grep -v grep | awk '{print \$2}'")

if [ -n "$RUNNING_PID" ]; then
    echo -e "${YELLOW}Found running training process (PID: $RUNNING_PID). Stopping it...${NC}"
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "kill $RUNNING_PID || sudo kill -9 $RUNNING_PID"
    echo -e "${GREEN}Process stopped.${NC}"
    sleep 2
fi

# Start the training with the fixed script - using updated precision format
echo -e "${BLUE}Starting training with v2 fixed script...${NC}"
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "cd /home/ubuntu && nohup python train_xlm_roberta_large_extended.py --train_manifest /home/ubuntu/ur_funny_train_humor_cleaned.csv --val_manifest /home/ubuntu/ur_funny_val_humor_cleaned.csv --model_name xlm-roberta-large --max_length 128 --batch_size 8 --learning_rate 1e-5 --epochs 15 --num_workers 4 --weight_decay 0.01 --dropout 0.1 --scheduler cosine --grad_clip 1.0 --log_dir training_logs_humor --exp_name xlm-roberta-large_extended_training --devices 1 --fp16 > training_output_v2.log 2>&1 &"

# Verify the process is running
sleep 5
RUNNING_PROCESS=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "ps aux | grep python | grep train_xlm_roberta_large_extended.py | grep -v grep")

if [ -z "$RUNNING_PROCESS" ]; then
    echo -e "${RED}Failed to start training process! Check the logs on the EC2 instance.${NC}"
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "cat training_output_v2.log"
    exit 1
else
    echo -e "${GREEN}Training process started successfully!${NC}"
    echo "$RUNNING_PROCESS"
    echo ""
    echo -e "${BLUE}To monitor the training, use:${NC}"
    echo -e "  ./monitor_xlm_roberta_extended_training.sh"
    echo ""
    echo -e "${BLUE}Or check the logs directly with:${NC}"
    echo -e "  ssh -i $SSH_KEY ubuntu@$EC2_IP 'tail -f /home/ubuntu/training_output_v2.log'"
fi
