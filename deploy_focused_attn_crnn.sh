#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem
export PROJECT="$PWD"

# Create local models directory if it doesn't exist
mkdir -p "$PROJECT/models"

echo "=========================================="
echo "Deploying Focused Attn-CRNN to EC2 GPU instance..."
echo "=========================================="
echo "EC2 instance: $IP"
echo "PEM key: $PEM"
echo "=========================================="

# Check if key file exists
if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    exit 1
fi

# Fix permissions on key
chmod 600 "$PEM"

# Prepare the specific files needed for training
echo "Creating minimal files directory for upload..."
mkdir -p /tmp/attn_crnn_upload/scripts

# Copy only the necessary files
cp scripts/train_attn_crnn.py /tmp/attn_crnn_upload/scripts/
# Add any other required dependencies here if needed
# cp scripts/other_dependency.py /tmp/attn_crnn_upload/scripts/

echo "Syncing only essential files to EC2..."
rsync -av -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
      /tmp/attn_crnn_upload/ ubuntu@$IP:~/emotion_project/

# Clean up temp directory
rm -rf /tmp/attn_crnn_upload

echo "Setting up training environment on EC2..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP << 'EOF'
    set -e

    # Create directories if they don't exist
    mkdir -p ~/emotion_project/models

    # Set up conda environment if not already set up
    if [ ! -d "$HOME/miniconda3" ]; then
        echo "Installing Miniconda..."
        wget -q -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3.sh -b -p $HOME/miniconda3
        rm Miniconda3.sh
    fi

    source ~/miniconda3/bin/activate

    # Create environment if it doesn't exist
    if ! conda info --envs | grep -q "emotion_env"; then
        echo "Creating emotion_env conda environment..."
        conda create -n emotion_env python=3.9 -y
    fi

    # Activate and install dependencies
    conda activate emotion_env
    echo "Installing TensorFlow and dependencies..."
    pip install tensorflow==2.15.0 tensorflow-addons seaborn tqdm matplotlib numpy pandas scikit-learn

    # Install system packages
    sudo apt-get update && sudo apt-get install -y tmux htop

    echo "Environment setup complete."
EOF

echo "Launching training job in tmux session..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP << 'EOF'
    source ~/miniconda3/bin/activate emotion_env
    cd ~/emotion_project

    # Check if tmux session already exists; if so kill it
    tmux has-session -t audio_train 2>/dev/null && tmux kill-session -t audio_train

    # Define the variable for the data directory and check if it exists
    DATA_DIR="/mnt/data/crema_d_features"
    if [ ! -d "$DATA_DIR" ]; then
        # Use current directory if data directory doesn't exist
        DATA_DIR="$PWD"
        echo "WARNING: /mnt/data/crema_d_features not found. Using $DATA_DIR instead."
    fi

    # Launch new tmux session with training command
    tmux new-session -d -s audio_train
    tmux send-keys -t audio_train "source ~/miniconda3/bin/activate emotion_env" C-m
    tmux send-keys -t audio_train "cd ~/emotion_project" C-m
    tmux send-keys -t audio_train "echo 'Starting Attn-CRNN training at $(date)'" C-m
    tmux send-keys -t audio_train "CUDA_VISIBLE_DEVICES=0 python scripts/train_attn_crnn.py --data_dirs $DATA_DIR --augment" C-m

    echo "Training job launched in tmux session 'audio_train'"
    echo "Use 'tmux attach -t audio_train' to view the progress"
    echo "Use Ctrl-b + d to detach from the session without stopping it"
EOF

echo "=========================================="
echo "Training job deployed successfully!"
echo "=========================================="
echo "To monitor training:"
echo "  ./monitor_attn_crnn_training.sh -l"
echo ""
echo "To check GPU status:"
echo "  ./monitor_attn_crnn_training.sh -s"
echo ""
echo "To download the trained model when complete:"
echo "  ./monitor_attn_crnn_training.sh -d"
echo "=========================================="
