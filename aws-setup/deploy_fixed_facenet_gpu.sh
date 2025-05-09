#!/bin/bash
# Deploy and run fixed Facenet LSTM training on GPU-enabled AWS EC2 instance
# This script handles deployment, setup, and monitoring for GPU-accelerated training

set -e

# Configuration
SSH_KEY="${SSH_KEY:-~/.ssh/aws-key.pem}"
INSTANCE_IP="${INSTANCE_IP:-}"
S3_BUCKET="${S3_BUCKET:-emotion-recognition-models}"
INSTANCE_TYPE="g5.xlarge"  # or p4d.24xlarge for more powerful training
MODEL_NAME="facenet_lstm_fixed_gpu"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
MODEL_DIR="facenet_lstm_${TIMESTAMP}"
REPO_DIR="/home/ubuntu/emotion-recognition"
VENV_DIR="/home/ubuntu/facenet-venv"
LOG_FILE="training_log_${TIMESTAMP}.txt"

# Parse command line args
INSTANCE_DEPLOYMENT=false
LOCAL_MODE=false
MONITORING_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --setup-instance)
      INSTANCE_DEPLOYMENT=true
      shift
      ;;
    --local)
      LOCAL_MODE=true
      shift
      ;;
    --monitor-only)
      MONITORING_ONLY=true
      shift
      ;;
    --instance)
      INSTANCE_IP="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --setup-instance    Setup a new instance (installs dependencies)"
      echo "  --local             Run training locally (for testing)"
      echo "  --monitor-only      Only monitor an existing training run"
      echo "  --instance IP       Specify EC2 instance IP"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$INSTANCE_IP" && "$LOCAL_MODE" == "false" ]]; then
  echo "Error: Instance IP is required for remote deployment."
  echo "Use --instance IP or set INSTANCE_IP environment variable."
  exit 1
fi

# Local execution mode
if [[ "$LOCAL_MODE" == "true" ]]; then
  echo "=== Running in local mode (for testing) ==="
  chmod +x deploy_fixed_facenet_training_complete.sh
  ./deploy_fixed_facenet_training_complete.sh --local
  exit 0
fi

# Check if SSH key exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo "Error: SSH key not found at $SSH_KEY"
  exit 1
fi

echo "=== Fixed Facenet Video-Only LSTM Training - GPU Deployment ==="
echo "Instance: $INSTANCE_IP"
echo "Instance type: $INSTANCE_TYPE"
echo "Model name: $MODEL_NAME"
echo "Timestamp: $TIMESTAMP"

if [[ "$MONITORING_ONLY" == "true" ]]; then
  echo "=== Monitoring Only Mode ==="
  ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "cd $REPO_DIR && tail -f nohup.out"
  exit 0
fi

# Setup instance (if requested)
if [[ "$INSTANCE_DEPLOYMENT" == "true" ]]; then
  echo "=== Setting up instance ==="
  
  # Upload instance setup script
  cat > instance_setup.sh << 'EOF'
#!/bin/bash
set -e

# Install required packages
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv build-essential \
    ffmpeg libsm6 libxext6 libxrender-dev nvidia-cuda-toolkit

# Create virtual environment
echo "Setting up Python environment..."
python3 -m venv ~/facenet-venv
source ~/facenet-venv/bin/activate

# Clone repo if it doesn't exist
if [ ! -d "~/emotion-recognition" ]; then
  git clone https://github.com/yourusername/emotion-recognition.git ~/emotion-recognition
else
  cd ~/emotion-recognition
  git pull
fi

# Install dependencies
cd ~/emotion-recognition
pip install --upgrade pip
pip install tensorflow[cuda] tensorflow-gpu numpy matplotlib pandas seaborn scikit-learn \
    opencv-python tqdm h5py tensorboard pydot graphviz boto3 awscli

# Create data directories
mkdir -p ~/emotion-recognition/ravdess_features_facenet
mkdir -p ~/emotion-recognition/crema_d_features_facenet
mkdir -p ~/emotion-recognition/models

# Setup AWS credentials (if needed)
mkdir -p ~/.aws
cat > ~/.aws/credentials << 'AWSEOF'
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
AWSEOF

echo "Instance setup complete!"
EOF

  chmod +x instance_setup.sh
  scp -i "$SSH_KEY" instance_setup.sh ubuntu@"$INSTANCE_IP":~/instance_setup.sh
  ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "chmod +x ~/instance_setup.sh && ~/instance_setup.sh"
  rm instance_setup.sh
fi

# Push latest code to instance
echo "=== Deploying code ==="
git archive --format=tar.gz HEAD -o latest.tar.gz

# Transfer code and data to instance
scp -i "$SSH_KEY" latest.tar.gz ubuntu@"$INSTANCE_IP":~/latest.tar.gz
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "mkdir -p $REPO_DIR && tar -xzf ~/latest.tar.gz -C $REPO_DIR"
rm latest.tar.gz

# If we have the features locally, bundle and send them
if [ -d "./crema_d_features_facenet" ]; then
  echo "Bundling and transferring feature files..."
  tar -czf features.tar.gz crema_d_features_facenet ravdess_features_facenet
  scp -i "$SSH_KEY" features.tar.gz ubuntu@"$INSTANCE_IP":~/features.tar.gz
  ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "tar -xzf ~/features.tar.gz -C $REPO_DIR && rm ~/features.tar.gz"
  rm features.tar.gz
fi

# Setup TensorBoard
cat > setup_tensorboard.sh << 'EOF'
#!/bin/bash
source ~/facenet-venv/bin/activate
cd ~/emotion-recognition
mkdir -p logs/tensorboard
nohup tensorboard --logdir=logs/tensorboard --port=6006 &
echo "TensorBoard started on port 6006"
EOF

scp -i "$SSH_KEY" setup_tensorboard.sh ubuntu@"$INSTANCE_IP":~/setup_tensorboard.sh
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "chmod +x ~/setup_tensorboard.sh && ~/setup_tensorboard.sh"
rm setup_tensorboard.sh

# Generate training launch script
cat > launch_training.sh << EOF
#!/bin/bash
source $VENV_DIR/bin/activate
cd $REPO_DIR

# Make sure script is executable
chmod +x deploy_fixed_facenet_training_complete.sh

# Setup monitoring tools
mkdir -p logs
nvidia-smi -l 60 > logs/gpu_stats_$TIMESTAMP.csv &
GPU_MONITOR_PID=\$!

# Enable access to the GPU
export CUDA_VISIBLE_DEVICES=0

# Start training
nohup ./deploy_fixed_facenet_training_complete.sh --remote-gpu > $LOG_FILE 2>&1 &
TRAINING_PID=\$!
echo \$TRAINING_PID > training.pid

echo "Training started with PID \$TRAINING_PID"
echo "GPU monitoring started with PID \$GPU_MONITOR_PID"
echo "Log file: $LOG_FILE"
echo "Run 'tail -f $LOG_FILE' to monitor training progress"
EOF

# Transfer and execute training script
scp -i "$SSH_KEY" launch_training.sh ubuntu@"$INSTANCE_IP":~/launch_training.sh
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "chmod +x ~/launch_training.sh && ~/launch_training.sh"
rm launch_training.sh

# Create monitoring script
cat > setup_monitoring.sh << EOF
#!/bin/bash
# Setup continuous monitoring for training progress

# Create progress visualization script
cat > ~/emotion-recognition/monitor_progress.py << 'PYEOF'
#!/usr/bin/env python3
import re
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_metrics(log_file):
    """Extract training and validation metrics from log file."""
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    epoch_pattern = r'Epoch (\d+)/\d+'
    metrics_pattern = r'loss: ([\d\.]+) - accuracy: ([\d\.]+) - val_loss: ([\d\.]+) - val_accuracy: ([\d\.]+) - lr: ([\d\.e\-]+)'
    
    for line in lines:
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        metrics_match = re.search(metrics_pattern, line)
        if metrics_match:
            metrics['epoch'].append(current_epoch)
            metrics['train_loss'].append(float(metrics_match.group(1)))
            metrics['train_acc'].append(float(metrics_match.group(2)))
            metrics['val_loss'].append(float(metrics_match.group(3)))
            metrics['val_acc'].append(float(metrics_match.group(4)))
            metrics['lr'].append(float(metrics_match.group(5)))
    
    return metrics

def plot_metrics(metrics, output_dir='.'):
    """Create plots of training metrics."""
    plt.figure(figsize=(12, 10))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epoch'], metrics['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(metrics['epoch'], metrics['val_acc'], 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Compute baseline accuracy (random) for 6-class problem
    baseline = 1/6
    plt.axhline(y=baseline, color='g', linestyle='--', label='Random Baseline (16.7%)')
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Training Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'training_progress_{timestamp}.png'))
    print(f"Plot saved to {os.path.join(output_dir, f'training_progress_{timestamp}.png')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_progress.py <log_file> [<output_dir>]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        sys.exit(1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics = extract_metrics(log_file)
    
    if not metrics['epoch']:
        print("No training metrics found in log file")
        sys.exit(0)
    
    print(f"Found {len(metrics['epoch'])} epochs of training data")
    print(f"Latest validation accuracy: {metrics['val_acc'][-1]:.4f}")
    print(f"Latest training accuracy: {metrics['train_acc'][-1]:.4f}")
    print(f"Latest validation loss: {metrics['val_loss'][-1]:.4f}")
    print(f"Learning rate: {metrics['lr'][-1]:.6f}")
    
    plot_metrics(metrics, output_dir)
PYEOF

# Create monitoring cron job
cat > ~/monitor_cron.sh << CRONEOF
#!/bin/bash
source $VENV_DIR/bin/activate
cd $REPO_DIR

# Check if training is still running
if [ -f "training.pid" ]; then
  PID=\$(cat training.pid)
  if ps -p \$PID > /dev/null; then
    # Training is running, update progress
    LOG_FILE="$LOG_FILE"
    if [ -f "\$LOG_FILE" ]; then
      python monitor_progress.py "\$LOG_FILE" "logs/progress_plots"
      
      # Extract best validation accuracy
      BEST_ACC=\$(grep -oP 'val_accuracy improved from .* to \K[\d\.]+' "\$LOG_FILE" | sort -nr | head -1)
      if [ ! -z "\$BEST_ACC" ]; then
        echo "\$(date): Best validation accuracy so far: \$BEST_ACC" >> logs/best_accuracy.log
      fi
      
      # Copy latest model to S3 bucket
      if [ -d "models/${MODEL_DIR}" ]; then
        aws s3 sync models/${MODEL_DIR} s3://${S3_BUCKET}/${MODEL_DIR} --exclude "*" --include "best_model.h5"
      fi
    fi
  fi
fi
CRONEOF

chmod +x ~/monitor_cron.sh

# Set up cron job to run every 30 minutes
(crontab -l 2>/dev/null; echo "*/30 * * * * ~/monitor_cron.sh >> ~/cron.log 2>&1") | crontab -

# Create monitoring script
cat > ~/monitor_training.sh << MONEOF
#!/bin/bash
# Interactive monitoring script
cd $REPO_DIR

echo "Monitoring GPU utilization:"
nvidia-smi

echo -e "\nMonitoring training log:"
tail -n 50 $LOG_FILE

echo -e "\nLatest validation results:"
grep -n "val_accuracy" $LOG_FILE | tail -n 3

if [ -f "logs/best_accuracy.log" ]; then
  echo -e "\nBest validation accuracy:"
  tail -n 1 logs/best_accuracy.log
fi

echo -e "\nFor continuous monitoring: 'tail -f $LOG_FILE'"
echo "To view TensorBoard: ssh -L 16006:localhost:6006 ubuntu@$INSTANCE_IP"
echo "Then open http://localhost:16006 in your browser"
MONEOF

chmod +x ~/monitor_training.sh
EOF

# Transfer and execute monitoring setup
scp -i "$SSH_KEY" setup_monitoring.sh ubuntu@"$INSTANCE_IP":~/setup_monitoring.sh
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "chmod +x ~/setup_monitoring.sh && ~/setup_monitoring.sh"
rm setup_monitoring.sh

echo "=== Training setup complete ==="
echo "Model training started on $INSTANCE_IP"
echo "Model: $MODEL_NAME"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor training:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP ~/monitor_training.sh"
echo ""
echo "To create a TensorBoard tunnel:"
echo "  ssh -L 16006:localhost:6006 -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo "  Then open http://localhost:16006 in your browser"
echo ""
echo "To stop training:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP \"pkill -f deploy_fixed_facenet_training_complete.sh\""
echo ""
echo "To download trained model:"
echo "  ./download_model.sh --instance $INSTANCE_IP --model $MODEL_DIR"

# Start monitoring immediately
if [[ "$INSTANCE_DEPLOYMENT" == "false" ]]; then
  echo "=== Live monitoring ==="
  ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "~/monitor_training.sh && tail -f $REPO_DIR/$LOG_FILE"
fi
