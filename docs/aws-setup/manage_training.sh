#!/bin/bash
# Master script to manage training job on AWS

# Set up colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"

# Make sure AWS credentials are set up
if [ ! -f ~/.aws/credentials ]; then
    echo -e "${YELLOW}AWS credentials not found. Setting them up now...${NC}"
    ./configure_aws.sh
fi

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}     Emotion Recognition Training Management        ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "${BLUE}Instance ID:${NC} $INSTANCE_ID"
echo -e "${BLUE}Instance IP:${NC} $INSTANCE_IP"
echo -e "${GREEN}====================================================${NC}"

echo -e "\n${YELLOW}Choose an action:${NC}"
echo "1) Check training progress"
echo "2) Monitor system resources"
echo "3) Upload updated training script"
echo "4) Download training results"
echo "5) Stop/Terminate instance"
echo "6) Exit"

read -p "Enter your choice [1-6]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Checking training progress...${NC}"
        ./check_progress.sh
        ;;
    2)
        echo -e "\n${YELLOW}Monitoring system resources...${NC}"
        ./monitor_cpu.sh
        ;;
    3)
        echo -e "\n${YELLOW}Preparing to upload updated training script...${NC}"
        echo "Please enter your username for the EC2 instance (default: ec2-user):"
        read -p "Username: " USERNAME
        USERNAME=${USERNAME:-ec2-user}
        
        echo "Copying fixed train_branched_6class.py with UTF-8 encoding declaration..."
        scp ../scripts/train_branched_6class.py ${USERNAME}@${INSTANCE_IP}:~/emotion_training/scripts/
        
        echo -e "${GREEN}File uploaded. Restarting training...${NC}"
        ssh ${USERNAME}@${INSTANCE_IP} "cd ~/emotion_training && nohup python scripts/train_branched_6class.py > training.log 2>&1 &"
        echo -e "${GREEN}Training restarted. Check progress with option 1.${NC}"
        ;;
    4)
        echo -e "\n${YELLOW}Downloading training results...${NC}"
        ./download_results.sh
        ;;
    5)
        echo -e "\n${YELLOW}Managing instance state...${NC}"
        ./stop_instance_updated.sh
        ;;
    6)
        echo -e "\n${GREEN}Exiting. Remember to terminate your instance when done to avoid additional charges.${NC}"
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Operation completed. Run this script again to perform another action.${NC}"
