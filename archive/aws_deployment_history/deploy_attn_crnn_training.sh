#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem
export PROJECT="$PWD"

# Create models directory if it doesn't exist
mkdir -p "$PROJECT/models"

echo "=========================================="
echo "Deploying Attn-CRNN to EC2 GPU instance..."
echo "=========================================="
echo "Project path: $PROJECT"
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

echo "Syncing minimal required files to EC2..."
rsync -av --exclude '.venv' --exclude '.git' \
      --exclude 'crema_d_features/' --exclude 'crema_d_features_facenet/' \
      --exclude 'ravdess_features/' --exclude 'ravdess_features_facenet/' \
      --exclude 'temp_resampled_videos/' --exclude 'emotion_training_logs/' \
      --exclude 'sample_wav2vec_data/' --exclude 'local_wav2vec_test_data/' \
      --exclude 'checkpoints/' --exclude 'models/' --exclude 'data/' \
      --exclude 'feature_verification_output/' --exclude 'feature_analysis_output/' \
      --exclude 'comparison_results/' --exclude 'comparison_results_ec2/' \
      --exclude 'analysis_output/' --exclude 'analysis_results/' \
      --exclude 'dataset_comparisons/' --exclude 'model_evaluation/' \
      --exclude 'npz_visualizations/' --exclude 'batch_analysis/' \
      -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
      "$PROJECT/scripts" "$PROJECT/requirements.txt" ubuntu@$IP:~/emotion_project/

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
    pip install tensorflow==2.15.0 tensorflow-addons seaborn tqdm
    
    # Install system packages
    sudo apt-get update && sudo apt-get install -y tmux htop
    
    echo "Environment setup complete."
EOF

echo "Launching training job in tmux session..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP << 'EOF'
    source ~/miniconda3/bin/activate emotion_env
    cd ~/emotion_project
    
    # Check if tmux session already exists; if so, kill it
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
echo "  ssh -i $PEM ubuntu@$IP"
echo "  tmux attach -t audio_train"
echo ""
echo "To check GPU status:"
echo "  ssh -i $PEM ubuntu@$IP 'nvidia-smi'"
echo ""
echo "To download the trained model when complete:"
echo "  scp -i $PEM ubuntu@$IP:~/emotion_project/best_attn_crnn_model.h5 $PROJECT/models/"
echo "=========================================="
