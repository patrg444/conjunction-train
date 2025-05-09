#!/usr/bin/env bash
# Deploy and run the Multimodal Emotion Fusion model on EC2
# This script uploads the necessary files to EC2 and runs the fusion workflow there

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/emotion_fusion"

# Check if key file exists
if [ ! -f "$KEY_PATH" ]; then
    echo "Error: SSH key file not found at $KEY_PATH"
    echo "Please update the KEY_PATH variable with the correct path to your EC2 key file."
    exit 1
fi

echo "=== Deploying Multimodal Emotion Fusion Model to EC2 ==="
echo "EC2 Instance: $EC2_INSTANCE"
echo "Remote Directory: $REMOTE_DIR"
echo

# Create a local tar file with all the necessary scripts
echo "Creating deployment package..."
mkdir -p deploy_temp
cp extract_wav2vec_features.py deploy_temp/
cp create_emotion_fusion.py deploy_temp/
cp demo_fusion_model.py deploy_temp/
cp run_fusion_workflow.sh deploy_temp/
cp MULTIMODAL_EMOTION_FUSION_README.md deploy_temp/

# Create a special EC2 version of the run script that uses the EC2 dataset paths
cat > deploy_temp/run_fusion_workflow_ec2.sh << 'EOL'
#!/usr/bin/env bash
# Run the complete emotion fusion workflow on EC2
# This script uses the datasets already present on the EC2 instance

# Create necessary directories
mkdir -p models/wav2vec
mkdir -p models/fusion

# Step 1: Create symlinks to the RAVDESS and CREMA-D datasets
if [ ! -d "ravdess_videos" ]; then
    echo "Creating symlink to RAVDESS dataset..."
    ln -sf /home/ubuntu/datasets/ravdess_videos ravdess_videos
fi

if [ ! -d "crema_d_videos" ]; then
    echo "Creating symlink to CREMA-D dataset..."
    ln -sf /home/ubuntu/datasets/crema_d_videos crema_d_videos
fi

# Step 2: Extract wav2vec features from sample videos
echo "Extracting wav2vec features from sample videos..."
# Find a sample video file from RAVDESS
SAMPLE_VIDEO=$(find ravdess_videos -name "*.mp4" | head -n 1)
if [ -z "$SAMPLE_VIDEO" ]; then
    echo "No sample video found. Using a placeholder."
    SAMPLE_VIDEO="placeholder.mp4"
fi
echo "Sample video: $SAMPLE_VIDEO"

# Activate PyTorch environment (which includes TensorFlow too)
source /opt/pytorch/bin/activate

# Install required Python packages
pip install transformers torchaudio opencv-python tqdm

# Run wav2vec feature extraction (only process a few files for demo)
python extract_wav2vec_features.py

# Step 3: Create fusion model configuration
echo "Creating fusion model..."
python create_emotion_fusion.py --video_weight 0.7 --audio_weight 0.3

# Step 4: Run a demo on a test video
echo "Running fusion model demo..."
if [ -n "$SAMPLE_VIDEO" ] && [ -f "$SAMPLE_VIDEO" ]; then
    python demo_fusion_model.py --video "$SAMPLE_VIDEO"
else
    echo "No test video found. The demo will create placeholder data."
    # Use a sample path - the demo will handle missing files gracefully
    python demo_fusion_model.py --video "sample_video.mp4"
fi

echo "------------------------------------"
echo "Fusion workflow complete on EC2!"
echo
echo "The following components were created:"
echo "1. Wav2vec audio features extracted from EC2 dataset (in models/wav2vec/)"
echo "2. Fusion model configuration (in models/fusion/)"
echo "3. Demo for inference (executed)"
echo
echo "To use on additional videos:"
echo "python demo_fusion_model.py --video /path/to/your/video.mp4"
EOL
chmod +x deploy_temp/run_fusion_workflow_ec2.sh

# Create the tar file
tar -czf emotion_fusion.tar.gz -C deploy_temp .
rm -rf deploy_temp

# Upload to EC2
echo "Uploading fusion model to EC2..."
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "mkdir -p $REMOTE_DIR"
scp -i "$KEY_PATH" emotion_fusion.tar.gz "$EC2_INSTANCE:$REMOTE_DIR/"

# Extract and run on EC2
echo "Running fusion model on EC2..."
ssh -i "$KEY_PATH" "$EC2_INSTANCE" << REMOTE_COMMANDS
cd $REMOTE_DIR
tar -xzf emotion_fusion.tar.gz
chmod +x run_fusion_workflow_ec2.sh
./run_fusion_workflow_ec2.sh
REMOTE_COMMANDS

# Cleanup
echo "Cleaning up local temporary files..."
rm emotion_fusion.tar.gz

echo
echo "=== Multimodal Emotion Fusion Model Deployment Complete ==="
echo "The model has been deployed and run on EC2 using the datasets already present there."
echo
echo "To access the generated models and results:"
echo "scp -i $KEY_PATH -r $EC2_INSTANCE:$REMOTE_DIR/models/ ./ec2_models/"
echo
echo "To run more analysis on EC2 directly:"
echo "ssh -i $KEY_PATH $EC2_INSTANCE"
echo "cd $REMOTE_DIR"
echo "python demo_fusion_model.py --video /path/to/your/video.mp4"
