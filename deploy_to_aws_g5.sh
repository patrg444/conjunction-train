#!/usr/bin/env bash
# Streamlined deployment script for the g5.2xlarge instance
# This directly targets the AWS instance at 52.90.38.179

# Set AWS IP address
AWS_IP="18.208.166.91"
EPOCHS=100
BATCH_SIZE=256
MAX_SEQ_LEN=45
LAUGH_WEIGHT=0.3

# Set SSH variables
SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$AWS_IP"

# Local and EC2 paths
LOCAL_MODEL_DIR="models/dynamic_padding_no_leakage"
EC2_HOME="/home/$SSH_USER"
EC2_PROJECT="emotion-recognition"
EC2_PROJECT_PATH="$EC2_HOME/$EC2_PROJECT"

# Create timestamp
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
echo "Deployment timestamp: $TIMESTAMP"

# Create archive of essential script files
echo "Creating script archive..."
tar -czf laughter_scripts.tar.gz \
    scripts/train_audio_pooling_lstm_with_laughter.py \
    scripts/audio_pooling_generator.py \
    scripts/feature_normalizer.py \
    run_audio_pooling_with_laughter.sh \
    scripts/demo_emotion_with_laughter.py \
    Makefile \
    requirements.txt

# Create archive of normalization statistics
echo "Creating normalization statistics archive..."
tar -czf normalization_stats.tar.gz $LOCAL_MODEL_DIR/*_normalization_stats.pkl

# Set up SSH connection and directory structure
echo "Setting up project structure on AWS..."
ssh -o StrictHostKeyChecking=no -i $SSH_KEY $SSH_HOST "mkdir -p $EC2_PROJECT_PATH/{scripts,models,datasets/manifests,logs}"

# Transfer files
echo "Transferring files to AWS..."
scp -i $SSH_KEY laughter_scripts.tar.gz $SSH_HOST:$EC2_HOME/
scp -i $SSH_KEY normalization_stats.tar.gz $SSH_HOST:$EC2_HOME/

# Extract files on AWS
echo "Extracting files on AWS..."
ssh -i $SSH_KEY $SSH_HOST "cd $EC2_PROJECT_PATH && \
    tar -xzf $EC2_HOME/laughter_scripts.tar.gz && \
    mkdir -p $EC2_PROJECT_PATH/$LOCAL_MODEL_DIR && \
    tar -xzf $EC2_HOME/normalization_stats.tar.gz -C $EC2_PROJECT_PATH/$LOCAL_MODEL_DIR && \
    chmod +x run_audio_pooling_with_laughter.sh scripts/*.py"

# Create the training launch script
echo "Creating training script..."
ssh -i $SSH_KEY $SSH_HOST "cat > $EC2_PROJECT_PATH/train_g5_${TIMESTAMP}.sh << 'EOF'
#!/usr/bin/env bash
cd $EC2_PROJECT_PATH

# Install dependencies
echo 'Installing requirements...'
pip install -r requirements.txt

# Create laughter manifest directory
mkdir -p datasets/manifests

# Run training with GPU-optimized settings
echo 'Starting training with batch size $BATCH_SIZE, epochs $EPOCHS...'
bash run_audio_pooling_with_laughter.sh $EPOCHS $BATCH_SIZE $MAX_SEQ_LEN $LAUGH_WEIGHT
EOF"

# Make training script executable
ssh -i $SSH_KEY $SSH_HOST "chmod +x $EC2_PROJECT_PATH/train_g5_${TIMESTAMP}.sh"

# Create download script
echo "Creating download script..."
cat > download_g5_model_${TIMESTAMP}.sh << EOF
#!/usr/bin/env bash
# Script to download the trained model from EC2
SSH_KEY="$SSH_KEY"
SSH_USER="$SSH_USER"
SSH_HOST="$SSH_USER@$AWS_IP"
EC2_PROJECT_PATH="$EC2_PROJECT_PATH"

# Find the model directory with timestamp
MODEL_DIR=\$(ssh -i \$SSH_KEY \$SSH_HOST "find \$EC2_PROJECT_PATH/models -name 'audio_pooling_with_laughter_*' -type d | sort -r | head -1")
if [ -z "\$MODEL_DIR" ]; then
    echo "Error: No model directory found on EC2"
    exit 1
fi

# Extract basename
MODEL_NAME=\$(basename "\$MODEL_DIR")
echo "Found model: \$MODEL_NAME"

# Create local directory
mkdir -p "models/\$MODEL_NAME"

# Download model files
echo "Downloading model files..."
scp -i \$SSH_KEY "\$SSH_HOST:\$MODEL_DIR/model_best.h5" "models/\$MODEL_NAME/"
scp -i \$SSH_KEY "\$SSH_HOST:\$MODEL_DIR/training_history.json" "models/\$MODEL_NAME/"
scp -i \$SSH_KEY "\$SSH_HOST:\$MODEL_DIR/model_info.json" "models/\$MODEL_NAME/"
scp -i \$SSH_KEY "\$SSH_HOST:\$MODEL_DIR/test_results.json" "models/\$MODEL_NAME/"

echo "Model downloaded to models/\$MODEL_NAME/"
EOF

chmod +x download_g5_model_${TIMESTAMP}.sh
echo "Created download script: download_g5_model_${TIMESTAMP}.sh"

# Display instructions
echo "======================================"
echo "Deployment complete!"
echo "======================================"
echo "To start training on the g5.2xlarge instance:"
echo "  ssh -i $SSH_KEY $SSH_HOST"
echo "  cd $EC2_PROJECT_PATH"
echo "  ./train_g5_${TIMESTAMP}.sh"
echo ""
echo "To download the trained model after completion:"
echo "  ./download_g5_model_${TIMESTAMP}.sh"
echo "======================================"

# Clean up local temporary files
rm laughter_scripts.tar.gz
rm normalization_stats.tar.gz

echo "Deployment completed at $(date)"
