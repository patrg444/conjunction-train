#!/bin/bash
# Simple script to deploy our Facenet files to GPU and run training

set -e

# Configuration
SSH_KEY="${SSH_KEY:-/Users/patrickgloria/Downloads/gpu-key.pem}"
INSTANCE_IP="${1:-18.208.166.91}"
REMOTE_DIR="/home/ubuntu/emotion-recognition"

if [[ -z "$INSTANCE_IP" ]]; then
  echo "Usage: $0 <instance-ip>"
  echo "Or set INSTANCE_IP environment variable"
  exit 1
fi

echo "=== Deploying Facenet pipeline to $INSTANCE_IP ==="
echo "SSH key: $SSH_KEY"
echo "Remote directory: $REMOTE_DIR"

# Check if SSH key exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo "Error: SSH key not found at $SSH_KEY"
  exit 1
fi

# Create tarball of required files
echo "Creating tarball of required files..."
# Make sure we explicitly include our fixed generator implementation
cp scripts/fixed_video_facenet_generator.py ./
tar -czf facenet_deploy.tar.gz fixed_video_facenet_generator.py scripts/*.py *.sh models/ crema_d_features_facenet/ ravdess_features_facenet/

# Create remote directory if it doesn't exist
echo "Setting up remote directory..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "mkdir -p $REMOTE_DIR"

# Upload tarball
echo "Uploading files..."
scp -i "$SSH_KEY" facenet_deploy.tar.gz ubuntu@"$INSTANCE_IP":~/facenet_deploy.tar.gz

# Extract on remote and setup
echo "Extracting files on remote..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "cd $REMOTE_DIR && tar -xzf ~/facenet_deploy.tar.gz && rm ~/facenet_deploy.tar.gz"

# Clean up local tarball
rm facenet_deploy.tar.gz

# Setup remote environment and start training
echo "Setting up environment and starting test training..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "cd $REMOTE_DIR && \
  sudo apt-get update && \
  sudo apt-get install -y python3-pip python3-venv tmux && \
  python3 -m venv ~/facenet-venv && \
  source ~/facenet-venv/bin/activate && \
  pip install --upgrade pip && \
  pip install tensorflow numpy matplotlib pandas scikit-learn opencv-python tqdm h5py && \
  tmux new-session -d -s facenet 'cd $REMOTE_DIR && \
  source ~/facenet-venv/bin/activate && \
  chmod +x run_facenet_tools.sh && \
  ./run_facenet_tools.sh test-train --batch-size 16'"

echo "=== Deployment completed ==="
echo "Training is running in tmux session 'facenet'"
echo ""
echo "To monitor training:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo "  tmux attach -t facenet"
echo ""
echo "To detach from tmux, press Ctrl+B then D"
