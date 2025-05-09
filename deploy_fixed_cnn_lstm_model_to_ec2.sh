#!/bin/bash

# This script deploys and runs the fixed CNN-LSTM model on the EC2 instance

# Configuration
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/emotion_project"

echo "Deploying fixed CNN-LSTM model to EC2 instance $EC2_IP..."

# Create temporary directory for deployment files
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Copy necessary files to temp directory
cp fix_cnn_lstm_model.py "$TEMP_DIR/"
cp run_fixed_cnn_lstm_model.sh "$TEMP_DIR/"
cp CNN_LSTM_FIX_README.md "$TEMP_DIR/"

# Create a server-specific version of the run script with correct paths
cat > "$TEMP_DIR/run_ec2_cnn_lstm_model.sh" << 'EOL'
#!/bin/bash

# Create necessary directories (already exist on EC2)
mkdir -p data/ravdess_features_cnn_fixed
mkdir -p data/crema_d_features_cnn_fixed
mkdir -p models

# Check model directories
if [ ! -L "data/ravdess_features_cnn_fixed" ] || [ ! -L "data/crema_d_features_cnn_fixed" ]; then
    echo "WARNING: CNN audio feature directories are symlinks. That's OK but just checking."
fi

# Set up environment
set -e  # Exit on any error
source ~/.bashrc

# Activate virtual environment if it exists
if [ -d "~/venv/bin/activate" ]; then
    source ~/venv/bin/activate
    echo "Activated virtual environment"
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the fixed model
echo "STARTING IMPROVED CNN-LSTM MODEL WITH DUAL ARCHITECTURE"
echo "======================================================="
echo "This script runs two models in parallel:"
echo "1. Bidirectional LSTM with temporal modeling"
echo "2. Global average pooling with dense layers"
echo "======================================================="

# Run the fixed model with GPU acceleration
python fix_cnn_lstm_model.py

# Check if the model ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Models trained and saved to models/cnn_lstm_fixed_* directory."
    
    # Check for saved model files
    LATEST_MODEL_DIR=$(find models -name "cnn_lstm_fixed_*" -type d | sort -r | head -n 1)
    if [ -n "$LATEST_MODEL_DIR" ]; then
        echo "Latest model directory: $LATEST_MODEL_DIR"
        echo "Files:"
        ls -la $LATEST_MODEL_DIR
    fi
else
    echo ""
    echo "Error occurred during model training. Check logs for more details."
    exit 1
fi
EOL

# Make run script executable
chmod +x "$TEMP_DIR/run_ec2_cnn_lstm_model.sh"

# Transfer files to EC2
echo "Transferring files to EC2..."
scp -i "$SSH_KEY" "$TEMP_DIR"/* ubuntu@$EC2_IP:$REMOTE_DIR/

# Make remote script executable
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "chmod +x $REMOTE_DIR/run_ec2_cnn_lstm_model.sh"

# Set up log file path
LOG_FILE="cnn_lstm_training_$(date +%Y%m%d_%H%M%S).log"

# Run the model on EC2 and log the output
echo "Running the model on EC2 instance (in background with nohup)..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "cd $REMOTE_DIR && nohup ./run_ec2_cnn_lstm_model.sh > $LOG_FILE 2>&1 &"

# Provide monitoring command
echo "Model training started on EC2 in background."
echo "To monitor training progress, use this command:"
echo "ssh -i \"$SSH_KEY\" ubuntu@$EC2_IP \"tail -f $REMOTE_DIR/$LOG_FILE\""

# Clean up
rm -rf "$TEMP_DIR"
echo "Deployment completed!"
