#!/bin/bash
# Script to fix GLIBCXX_3.4.29 issue on EC2 instance

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

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     FIXING GLIBCXX_3.4.29 ISSUE ON EC2 INSTANCE                 ${NC}"
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

# Create fix script to run on the instance
echo -e "${YELLOW}Creating fix script for GLIBCXX issue...${NC}"
cat > aws-setup/glibcxx_fix.sh << 'EOF'
#!/bin/bash
# Script to fix GLIBCXX_3.4.29 issue

# Install the development tools
sudo yum group install -y "Development Tools"

# Install centos-release-scl repository
sudo yum install -y centos-release-scl

# Install devtoolset-11
sudo yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++

# Create directory for the newer libstdc++
mkdir -p ~/lib

# Enable the devtoolset-11
source /opt/rh/devtoolset-11/enable

# Copy the newer libstdc++ libraries to our custom directory
cp /opt/rh/devtoolset-11/root/usr/lib/gcc/x86_64-redhat-linux/11/libstdc++.so.6* ~/lib/

# Check if the newer version of GLIBCXX is now available in the copied library
strings ~/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29
if [ $? -eq 0 ]; then
    echo "GLIBCXX_3.4.29 is available in the new library"
else
    echo "GLIBCXX_3.4.29 is still not available. Further troubleshooting required."
    exit 1
fi

# Create a wrapper script to run Python with the new library
cat > ~/emotion_training/run_with_new_libstdcxx.sh << 'INNEREOF'
#!/bin/bash
# Script to run Python with the newer libstdc++

# Set the LD_LIBRARY_PATH to use our custom libstdc++ before the system one
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

# Activate the conda environment
source $HOME/miniconda/bin/activate emotion_model

# Set the Python path to include the emotion_training directory
export PYTHONPATH=$HOME/emotion_training:$PYTHONPATH

# Go to the emotion_training directory
cd $HOME/emotion_training

# Run the training script with the newer libstdc++
echo "Starting LSTM attention model training at $(date)"
echo "Using libstdc++ from: $HOME/lib"
echo "Checking for GLIBCXX_3.4.29:"
strings $HOME/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

# Install TensorFlow with pip instead of conda
pip install tensorflow==2.9.0 scipy==1.10.1

# Run the training script
python scripts/train_branched_attention.py 2>&1 | tee lstm_attention_training.log
echo "Training completed at $(date) with exit code $?"
INNEREOF

chmod +x ~/emotion_training/run_with_new_libstdcxx.sh

# Kill any existing training sessions
pkill -f train_branched_attention.py || true
tmux kill-session -t lstm_attention 2>/dev/null || true

# Launch new training in tmux with the fixed library
tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./run_with_new_libstdcxx.sh"

echo "LSTM attention model training launched with fixed GLIBCXX library"
EOF

# Upload the script to the EC2 instance
echo -e "${YELLOW}Uploading fix script to EC2 instance...${NC}"
scp -i $KEY_FILE aws-setup/glibcxx_fix.sh ec2-user@$INSTANCE_IP:~/glibcxx_fix.sh

# Execute the script on the EC2 instance
echo -e "${YELLOW}Running fix script on EC2 instance...${NC}"
ssh -i $KEY_FILE ec2-user@$INSTANCE_IP "chmod +x ~/glibcxx_fix.sh && ~/glibcxx_fix.sh"

echo -e "${GREEN}GLIBCXX issue fix attempted.${NC}"
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
