#!/bin/bash
# This script prepares the emotion training on the remote instance

# Transfer the optimization script
scp -i "emotion-recognition-key-20250322081419.pem" -o StrictHostKeyChecking=no cpu_setup.sh ec2-user@3.235.66.190:~/

# Execute setup on the remote instance
ssh -i "emotion-recognition-key-20250322081419.pem" -o StrictHostKeyChecking=no ec2-user@3.235.66.190 << 'ENDSSH'
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
