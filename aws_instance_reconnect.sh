#!/bin/bash
# Script to connect to AWS instance after resize/restart

# Configuration
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
INSTANCE_IP="18.208.166.91"  # Current IP, update if changed

# Function to check if instance is responding
check_connection() {
  echo "Checking connection to $INSTANCE_IP..."
  timeout 5 ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$INSTANCE_IP "echo 'Connection successful'" > /dev/null 2>&1
  return $?
}

# Function to deploy training scripts with GPU device selection
deploy_training() {
  echo "Deploying training jobs on separate GPUs..."
  
  # Create deployment script for Facenet video training on GPU 0
  cat > deploy_facenet_gpu0.sh << 'INNER'
#!/bin/bash
cd /home/ubuntu/emotion-recognition
echo "Starting Facenet training on GPU 0 at $(date)"
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/train_video_only_facenet_lstm_key_fixed.py > video_only_facenet_lstm_key_fixed_gpu0.log 2>&1 &
echo "Job started with PID $!"
INNER
  
  # Create deployment script for Audio training on GPU 1
  cat > deploy_audio_gpu1.sh << 'INNER'
#!/bin/bash
cd /home/ubuntu/emotion-recognition
echo "Starting Audio training on GPU 1 at $(date)"
export CUDA_VISIBLE_DEVICES=1
nohup python scripts/train_audio_only_cnn_lstm_v2.py > audio_only_training_gpu1.log 2>&1 &
echo "Job started with PID $!"
INNER
  
  # Copy deployment scripts to server
  scp -i $SSH_KEY deploy_facenet_gpu0.sh deploy_audio_gpu1.sh ubuntu@$INSTANCE_IP:/home/ubuntu/
  
  # Make them executable
  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP "chmod +x /home/ubuntu/deploy_facenet_gpu0.sh /home/ubuntu/deploy_audio_gpu1.sh"
  
  # Clean up local files
  rm deploy_facenet_gpu0.sh deploy_audio_gpu1.sh
}

# Check EC2 instance setup
verify_instance() {
  echo "===== VERIFYING INSTANCE CONFIGURATION ====="
  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP "
    echo '- Checking instance type:'
    curl -s http://169.254.169.254/latest/meta-data/instance-type
    
    echo -e '\n- Checking available GPUs:'
    nvidia-smi -L
    
    echo -e '\n- Checking GPU memory:'
    nvidia-smi
    
    echo -e '\n- Checking system resources:'
    free -h
    df -h /
  "
}

# Main execution
echo "AWS EC2 Instance Reconnection Tool"
echo "=================================="
echo "This script will help reconnect to the EC2 instance after resizing"
echo "Current instance IP: $INSTANCE_IP"

# Try to connect
if check_connection; then
  echo "Successfully connected to instance!"
  verify_instance
  
  read -p "Do you want to deploy training scripts to separate GPUs? (y/n): " deploy_response
  if [[ $deploy_response == "y" ]]; then
    deploy_training
    echo "Training scripts deployed!"
  else
    echo "Training scripts not deployed. You can run them manually on the instance."
  fi
else
  echo "Connection to $INSTANCE_IP failed."
  echo ""
  echo "If the instance IP has changed, please update it and retry."
  echo "If the instance is still stopping/starting, please wait and retry."
  echo ""
  echo "To use this script with a different IP address:"
  echo "  1. Edit this script to update the INSTANCE_IP variable"
  echo "  2. Run the script again"
fi

echo "Done!"
