#!/bin/bash
# Script to connect to the launched EC2 instance

# Set up colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_ID="i-0f768453523079bd9"
INSTANCE_IP="3.235.66.190"
KEY_FILE="emotion-recognition-key-20250322081419.pem"

# Check if key file exists
if [ ! -f "$KEY_FILE" ] && [ -f "aws-setup/$KEY_FILE" ]; then
    KEY_FILE="aws-setup/$KEY_FILE"
fi

echo -e "${YELLOW}Connecting to instance $INSTANCE_ID at $INSTANCE_IP...${NC}"
echo -e "${YELLOW}Using key file: $KEY_FILE${NC}"

# Create setup scripts
echo -e "${YELLOW}Creating utility scripts for the instance...${NC}"

# CPU setup script
cat > cpu_setup.sh << 'CPUEOF'
#!/bin/bash
# CPU-specific optimizations for c5.24xlarge

# Configure TensorFlow for optimal CPU performance
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=96
export OMP_NUM_THREADS=96
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1

# For Intel CPUs (c5 instances use Intel CPUs)
export TF_ENABLE_ONEDNN_OPTS=1

# Disable CUDA to ensure CPU usage
export CUDA_VISIBLE_DEVICES=""
CPUEOF

# Create a script to check training progress
cat > check_progress.sh << EOF2
#!/bin/bash
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tail -f ~/emotion_training/training.log"
EOF2
chmod +x check_progress.sh

# Create a script to monitor CPU usage
cat > monitor_cpu.sh << EOF3
#!/bin/bash
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "top -b -n 1"
EOF3
chmod +x monitor_cpu.sh

# Create a script to download results
cat > download_results.sh << EOF4
#!/bin/bash
mkdir -p results
scp -i "$KEY_FILE" -r ec2-user@$INSTANCE_IP:~/emotion_training/models/branched_6class results/
echo "Results downloaded to results/branched_6class"
EOF4
chmod +x download_results.sh

# Create a script to stop/terminate the instance
cat > stop_instance.sh << EOF5
#!/bin/bash
# This script stops or terminates the EC2 instance

echo "Do you want to stop or terminate the instance?"
echo "1) Stop instance (can be restarted later, storage charges apply)"
echo "2) Terminate instance (permanent deletion, no further charges)"
read -p "Enter choice [1-2]: " choice

case \$choice in
    1)
        echo "Stopping instance $INSTANCE_ID..."
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
        echo "Instance stopped. To restart it later use: aws ec2 start-instances --instance-ids $INSTANCE_ID"
        ;;
    2)
        echo "Terminating instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
        echo "Instance terminated."
        ;;
    *)
        echo "Invalid choice, no action taken."
        ;;
esac
EOF5
chmod +x stop_instance.sh

# Create a setup script for the remote instance
cat > setup_remote.sh << EOF6
#!/bin/bash
# This script prepares the emotion training on the remote instance

# Transfer the optimization script
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no cpu_setup.sh ec2-user@$INSTANCE_IP:~/

# Execute setup on the remote instance
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'ENDSSH'
# Create directories for emotion training
mkdir -p ~/emotion_training

# Wait for instance to complete initialization and update packages
echo "Updating system packages..."
sudo yum update -y

# Install necessary packages
echo "Installing required packages..."
sudo yum install -y git gcc gcc-c++ make

# Clone the repository or set up your code manually
echo "Setting up the emotion recognition project..."
cd ~/emotion_training
git clone https://github.com/yourusername/emotion-recognition.git .

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Source the CPU optimizations to ./train.sh
echo "source ~/cpu_setup.sh" > train_wrapper.sh
echo "./train.sh" >> train_wrapper.sh
chmod +x train_wrapper.sh

# Start training with default parameters
echo "Starting training..."
nohup ./train_wrapper.sh > training.log 2>&1 &
echo "Training started in background. You can check progress with: tail -f training.log"
ENDSSH
EOF6
chmod +x setup_remote.sh

echo -e "${GREEN}Scripts created successfully.${NC}"
echo -e "${YELLOW}Available commands:${NC}"
echo "- ./setup_remote.sh - Set up and start the emotion recognition training on the instance"
echo "- ./check_progress.sh - Check training progress"
echo "- ./monitor_cpu.sh - Monitor CPU utilization in real-time"
echo "- ./download_results.sh - Download trained models"
echo "- ./stop_instance.sh - Stop or terminate the instance when done"
echo ""
echo -e "${RED}IMPORTANT: Remember to stop or terminate your instance when done to avoid unnecessary charges.${NC}"
echo -e "${RED}Hourly cost for c5.24xlarge is approximately \$4.08/hour.${NC}"
