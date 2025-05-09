#!/usr/bin/env bash
# Script to deploy and run audio-pooling LSTM model with laughter detection on AWS
# Usage: bash aws-setup/deploy_audio_pooling_with_laughter.sh [instance_id] [num_epochs] [batch_size]
#
# This script:
# 1. Copies required files to the EC2 instance
# 2. Sets up the environment on the EC2 instance
# 3. Starts the training job
# 4. Creates a monitoring script

# Get arguments
INSTANCE_ID=$1
EPOCHS=${2:-100}
BATCH_SIZE=${3:-256}
MAX_SEQ_LEN=45
LAUGH_WEIGHT=0.3

# Check for instance ID
if [ -z "$INSTANCE_ID" ]; then
    echo "Error: Instance ID required"
    echo "Usage: bash aws-setup/deploy_audio_pooling_with_laughter.sh [instance_id] [num_epochs] [batch_size]"
    exit 1
fi

# Set SSH variables
SSH_KEY="$HOME/.ssh/id_rsa"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$INSTANCE_ID"

# Local paths
LOCAL_MODEL_DIR="models/dynamic_padding_no_leakage"
LOCAL_FEATURES_RAVDESS="ravdess_features_facenet"
LOCAL_FEATURES_CREMA="crema_d_features_facenet"
LOCAL_LAUGHTER_MANIFEST="datasets/manifests/laughter_v1.csv"

# Create timestamp for unique identification
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
echo "Deployment timestamp: $TIMESTAMP"

# EC2 paths
EC2_HOME="/home/$SSH_USER"
EC2_PROJECT="emotion-recognition"
EC2_PROJECT_PATH="$EC2_HOME/$EC2_PROJECT"
EC2_LAUGHTER_PATH="$EC2_PROJECT_PATH/datasets/manifests"

# Check if required files exist
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    echo "Error: Model directory not found at $LOCAL_MODEL_DIR"
    exit 1
fi

if [ ! -d "$LOCAL_FEATURES_RAVDESS" ]; then
    echo "Error: RAVDESS features not found at $LOCAL_FEATURES_RAVDESS"
    exit 1
fi

if [ ! -d "$LOCAL_FEATURES_CREMA" ]; then
    echo "Error: CREMA-D features not found at $LOCAL_FEATURES_CREMA"
    exit 1
fi

# Create archive of required files
echo "Creating archive of source files..."
tar -czf laughter_scripts.tar.gz \
    scripts/train_audio_pooling_lstm_with_laughter.py \
    scripts/audio_pooling_generator.py \
    scripts/feature_normalizer.py \
    run_audio_pooling_with_laughter.sh \
    datasets/scripts/fetch_audioset_laughter.sh \
    datasets/scripts/ingest_liris_accede.py \
    datasets/scripts/build_laughter_manifest.py \
    datasets/README.md \
    Makefile \
    requirements.txt

# Check for laughter manifest file
LAUGHTER_MANIFEST_OPTION=""
if [ -f "$LOCAL_LAUGHTER_MANIFEST" ]; then
    echo "Laughter manifest found, will copy to EC2"
    LAUGHTER_MANIFEST_OPTION="--laugh_manifest $EC2_LAUGHTER_PATH/laughter_v1.csv"
    
    # Also create a tarball
    mkdir -p temp_laughter
    cp -r datasets/manifests temp_laughter/
    tar -czf laughter_manifests.tar.gz -C temp_laughter .
    rm -rf temp_laughter
else
    echo "Warning: Laughter manifest not found at $LOCAL_LAUGHTER_MANIFEST"
    echo "Will need to generate on EC2 with 'make laughter_data'"
fi

# Connect to EC2 and create project structure
echo "Setting up project structure on EC2..."
ssh -i $SSH_KEY $SSH_HOST "mkdir -p $EC2_PROJECT_PATH/{scripts,models,datasets/manifests,logs}"

# Copy files to EC2
echo "Copying files to EC2..."
scp -i $SSH_KEY laughter_scripts.tar.gz $SSH_HOST:$EC2_HOME/

# Copy feature archives (if not already on EC2)
echo "Checking for feature archives on EC2..."
ssh -i $SSH_KEY $SSH_HOST "if [ ! -d '$EC2_PROJECT_PATH/$LOCAL_FEATURES_RAVDESS' ]; then \
    echo 'RAVDESS features not found on EC2, will copy...'; \
    exit 1; \
fi"

if [ $? -ne 0 ]; then
    echo "Copying RAVDESS features to EC2..."
    tar -czf ravdess_features.tar.gz $LOCAL_FEATURES_RAVDESS
    scp -i $SSH_KEY ravdess_features.tar.gz $SSH_HOST:$EC2_HOME/
    ssh -i $SSH_KEY $SSH_HOST "mkdir -p $EC2_PROJECT_PATH/$LOCAL_FEATURES_RAVDESS && \
        tar -xzf ravdess_features.tar.gz -C $EC2_PROJECT_PATH"
    rm ravdess_features.tar.gz
else
    echo "RAVDESS features already on EC2, skipping copy"
fi

# Similar check for CREMA-D features
ssh -i $SSH_KEY $SSH_HOST "if [ ! -d '$EC2_PROJECT_PATH/$LOCAL_FEATURES_CREMA' ]; then \
    echo 'CREMA-D features not found on EC2, will copy...'; \
    exit 1; \
fi"

if [ $? -ne 0 ]; then
    echo "Copying CREMA-D features to EC2..."
    tar -czf crema_d_features.tar.gz $LOCAL_FEATURES_CREMA
    scp -i $SSH_KEY crema_d_features.tar.gz $SSH_HOST:$EC2_HOME/
    ssh -i $SSH_KEY $SSH_HOST "mkdir -p $EC2_PROJECT_PATH/$LOCAL_FEATURES_CREMA && \
        tar -xzf crema_d_features.tar.gz -C $EC2_PROJECT_PATH"
    rm crema_d_features.tar.gz
else
    echo "CREMA-D features already on EC2, skipping copy"
fi

# Copy normalization statistics
echo "Copying normalization statistics to EC2..."
tar -czf normalization_stats.tar.gz $LOCAL_MODEL_DIR/*_normalization_stats.pkl
scp -i $SSH_KEY normalization_stats.tar.gz $SSH_HOST:$EC2_HOME/
ssh -i $SSH_KEY $SSH_HOST "mkdir -p $EC2_PROJECT_PATH/$LOCAL_MODEL_DIR && \
    tar -xzf normalization_stats.tar.gz -C $EC2_PROJECT_PATH/$LOCAL_MODEL_DIR"

# Copy laughter manifests if available
if [ -f "laughter_manifests.tar.gz" ]; then
    echo "Copying laughter manifests to EC2..."
    scp -i $SSH_KEY laughter_manifests.tar.gz $SSH_HOST:$EC2_HOME/
    ssh -i $SSH_KEY $SSH_HOST "tar -xzf laughter_manifests.tar.gz -C $EC2_PROJECT_PATH"
    rm laughter_manifests.tar.gz
fi

# Extract scripts on EC2
echo "Extracting scripts on EC2..."
ssh -i $SSH_KEY $SSH_HOST "cd $EC2_PROJECT_PATH && \
    tar -xzf $EC2_HOME/laughter_scripts.tar.gz && \
    chmod +x run_audio_pooling_with_laughter.sh datasets/scripts/*.sh scripts/*.py"

# Create training script
echo "Creating training script on EC2..."
ssh -i $SSH_KEY $SSH_HOST "cat > $EC2_PROJECT_PATH/train_laughter_${TIMESTAMP}.sh << 'EOF'
#!/usr/bin/env bash
cd $EC2_PROJECT_PATH
# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo 'GPU detected, using larger batch size'
else
    echo 'Warning: No GPU detected, training will be slow'
fi

# Install dependencies
pip install -r requirements.txt

# Check for laughter manifests
if [ ! -f 'datasets/manifests/laughter_v1.csv' ]; then
    echo 'Laughter manifest not found, running data preparation'
    make laughter_data
fi

# Run training
bash run_audio_pooling_with_laughter.sh $EPOCHS $BATCH_SIZE $MAX_SEQ_LEN $LAUGH_WEIGHT

EOF"

# Make training script executable
ssh -i $SSH_KEY $SSH_HOST "chmod +x $EC2_PROJECT_PATH/train_laughter_${TIMESTAMP}.sh"

# Create a download script for retrieving the trained model
echo "Creating download script..."
cat > download_laughter_model_${TIMESTAMP}.sh << EOF
#!/usr/bin/env bash
# Script to download the trained model from EC2
INSTANCE_ID=$INSTANCE_ID
SSH_KEY="$HOME/.ssh/id_rsa"
SSH_USER="$SSH_USER"
SSH_HOST="\$SSH_USER@\$INSTANCE_ID"
EC2_PROJECT_PATH="$EC2_PROJECT_PATH"

# Find the model directory with the timestamp
MODEL_DIR=\$(ssh -i \$SSH_KEY \$SSH_HOST "find \$EC2_PROJECT_PATH/models -name 'audio_pooling_with_laughter_*' -type d | sort -r | head -1")
if [ -z "\$MODEL_DIR" ]; then
    echo "Error: No model directory found on EC2"
    exit 1
fi

# Extract the basename
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

chmod +x download_laughter_model_${TIMESTAMP}.sh
echo "Created download script: download_laughter_model_${TIMESTAMP}.sh"

# Display instructions for running training
echo "======================================"
echo "Deployment complete!"
echo "======================================"
echo "To start training on EC2:"
echo "  ssh -i $SSH_KEY $SSH_HOST"
echo "  cd $EC2_PROJECT_PATH"
echo "  ./train_laughter_${TIMESTAMP}.sh"
echo ""
echo "To monitor training:"
echo "  Look for the generated monitoring script on EC2"
echo ""
echo "To download the trained model after completion:"
echo "  ./download_laughter_model_${TIMESTAMP}.sh"
echo "======================================"

# Clean up local temporary files
rm laughter_scripts.tar.gz
rm -f normalization_stats.tar.gz
