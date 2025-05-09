#!/usr/bin/env bash
# Script to extract wav2vec features from the complete RAVDESS and CREMA-D datasets
# and download the resulting features to the local machine

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
LOCAL_FEATURES_DIR="./wav2vec_extracted_features"

echo "=== Processing Complete RAVDESS and CREMA-D Datasets ==="
echo "EC2 Instance: $EC2_INSTANCE"
echo "Remote Directory: $REMOTE_DIR"
echo

# Create a script to process all files
cat > process_all_files.sh << 'EOL'
#!/bin/bash
# Process all files in both datasets without sample limitation

# Activate the PyTorch environment
source /opt/pytorch/bin/activate

# Set the directory
cd /home/ubuntu/audio_emotion

# Ensure directories exist
mkdir -p models/wav2vec
mkdir -p models/fusion
mkdir -p temp_audio

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found, installing..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

echo "=== Starting full datasets extraction $(date) ==="
echo "This may take several hours depending on the number of files"

# Run extraction on all files (explicitly setting num_samples to 0 to process ALL files)
python fixed_extract_wav2vec.py \
  --ravdess_dir /home/ubuntu/datasets/ravdess_videos \
  --cremad_dir /home/ubuntu/datasets/crema_d_videos \
  --output_dir models/wav2vec \
  --temp_dir temp_audio \
  --num_samples 0

echo
echo "=== Full extraction completed at $(date) ==="
echo "Total files in output directory:"
find models/wav2vec -name "*.npz" | wc -l

# Create a tarball of the features for easier download
echo "Creating tarball of features..."
tar -czvf wav2vec_features.tar.gz models/wav2vec/
echo "Tarball created at: /home/ubuntu/audio_emotion/wav2vec_features.tar.gz"
EOL
chmod +x process_all_files.sh

# Upload the script
echo "Uploading processing script..."
scp -i "$KEY_PATH" process_all_files.sh "$EC2_INSTANCE:$REMOTE_DIR/"

# Execute remotely
echo "Starting full dataset extraction on EC2..."
echo "This will run in the background and may take several hours."
echo "No need to keep this terminal open - the process will continue on the server."
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "cd $REMOTE_DIR && nohup ./process_all_files.sh > full_extraction.log 2>&1 &"

echo
echo "Process started on EC2 in background mode."
echo "You can check progress using:"
echo "  ssh -i $KEY_PATH $EC2_INSTANCE \"cat $REMOTE_DIR/full_extraction.log\""
echo
echo "When complete, download results using:"
echo "  mkdir -p $LOCAL_FEATURES_DIR"
echo "  scp -i $KEY_PATH $EC2_INSTANCE:$REMOTE_DIR/wav2vec_features.tar.gz $LOCAL_FEATURES_DIR/"
echo "  tar -xzvf $LOCAL_FEATURES_DIR/wav2vec_features.tar.gz -C $LOCAL_FEATURES_DIR/"
echo
echo "To check the status of the extraction process:"
echo "  ssh -i $KEY_PATH $EC2_INSTANCE \"ps aux | grep fixed_extract_wav2vec.py\""
