#!/bin/bash
# Script to fix GLIBCXX_3.4.29 issue on Amazon Linux 2 EC2 instance

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
echo -e "${BLUE}     FIXING LIBSTDC++ ISSUE ON AMAZON LINUX 2 INSTANCE           ${NC}"
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
echo -e "${YELLOW}Creating fix script for Amazon Linux 2...${NC}"
cat > aws-setup/amazon_linux_fix.sh << 'EOF'
#!/bin/bash
# Fix GLIBCXX_3.4.29 issue on Amazon Linux 2

# Install GCC 9 from Amazon Linux repositories
echo "Installing GCC 9 from Amazon Linux Extra repository..."
sudo amazon-linux-extras install -y gcc9

# Install development tools
sudo yum group install -y "Development Tools"
sudo yum install -y gcc9-c++

# Setup a directory for the newer libraries
mkdir -p ~/custom_lib

# Compile a newer libstdc++ from source
echo "Building libstdc++ compatible with GLIBCXX_3.4.29..."
cd ~
if [ ! -d "gcc-10.2.0" ]; then
    wget https://ftp.gnu.org/gnu/gcc/gcc-10.2.0/gcc-10.2.0.tar.gz
    tar -xzf gcc-10.2.0.tar.gz
    cd gcc-10.2.0
    ./contrib/download_prerequisites
    cd ~
fi

cd gcc-10.2.0
mkdir -p build
cd build

# Configure and build only libstdc++
../configure --prefix=$HOME/gcc-10-libs --disable-multilib --enable-languages=c,c++ --disable-bootstrap
make -j$(nproc) all-target-libstdc++-v3
make install-target-libstdc++-v3

# Copy the libstdc++ library to our custom directory
cp ~/gcc-10-libs/lib64/libstdc++.so.6* ~/custom_lib/

# Verify the GLIBCXX version
echo "Verifying GLIBCXX versions in the newly built library:"
strings ~/custom_lib/libstdc++.so.6 | grep GLIBCXX

# Create a wrapper script to use the new library with Python
cat > ~/emotion_training/run_with_fixed_libs.sh << 'INNEREOF'
#!/bin/bash
# Run Python with the newer libstdc++ library

# Set LD_LIBRARY_PATH to use our custom libstdc++ library first
export LD_LIBRARY_PATH=$HOME/custom_lib:$LD_LIBRARY_PATH

# Activate conda environment
source $HOME/miniconda/bin/activate emotion_model

# Setup Python path for our project
export PYTHONPATH=$HOME/emotion_training:$PYTHONPATH

# Change to the emotion_training directory
cd $HOME/emotion_training

# Install libraries with pip (not conda) which avoids some dependency conflicts
pip install tensorflow==2.8.0
pip install scipy matplotlib pandas scikit-learn h5py

# Run the training script with the newer libstdc++
echo "Starting LSTM attention model training at $(date)"
echo "Using custom libstdc++ from: $HOME/custom_lib"
echo "Checking for GLIBCXX_3.4.29:"
strings $HOME/custom_lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

python scripts/train_branched_attention.py 2>&1 | tee lstm_attention_training.log
echo "Training completed at $(date) with exit code $?"
INNEREOF

chmod +x ~/emotion_training/run_with_fixed_libs.sh

# Kill any existing training sessions
pkill -f train_branched_attention.py || true
tmux kill-session -t lstm_attention 2>/dev/null || true

# Launch new training in tmux with the fixed library
tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./run_with_fixed_libs.sh"

echo "LSTM attention model training launched with fixed libraries"
echo "You can check the logs with: 'tmux attach -t lstm_attention'"
EOF

# Upload the script to the EC2 instance
echo -e "${YELLOW}Uploading fix script to EC2 instance...${NC}"
scp -i $KEY_FILE aws-setup/amazon_linux_fix.sh ec2-user@$INSTANCE_IP:~/amazon_linux_fix.sh

# Execute the script on the EC2 instance
echo -e "${YELLOW}Running fix script on EC2 instance...${NC}"
ssh -i $KEY_FILE ec2-user@$INSTANCE_IP "chmod +x ~/amazon_linux_fix.sh && ~/amazon_linux_fix.sh"

echo -e "${GREEN}Amazon Linux 2 libstdc++ fix attempted.${NC}"
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
