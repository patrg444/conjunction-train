#!/usr/bin/env bash
# Deploy and run the enhanced wav2vec extraction script on EC2
# This version supports multiple file formats (MP4, FLV, etc.) for CREMA-D dataset

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "=== Deploying Enhanced wav2vec Audio Extraction to EC2 ==="
echo "EC2 Instance: $EC2_INSTANCE"
echo "Remote Directory: $REMOTE_DIR"
echo

# Create remote directory if it doesn't exist
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "mkdir -p $REMOTE_DIR"

# Upload the enhanced script
echo "Uploading enhanced wav2vec extraction script..."
scp -i "$KEY_PATH" fixed_extract_wav2vec.py "$EC2_INSTANCE:$REMOTE_DIR/"

# Create and upload a runner script
cat > run_enhanced_wav2vec_extraction.sh << 'EOL'
#!/bin/bash
# Run the enhanced wav2vec extraction script on EC2

# Activate the PyTorch environment
source /opt/pytorch/bin/activate

# Make directories
mkdir -p models/wav2vec
mkdir -p models/fusion
mkdir -p temp_audio

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found, installing..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

# Run the extraction script with a limited number of samples for testing
python fixed_extract_wav2vec.py \
  --ravdess_dir /home/ubuntu/datasets/ravdess_videos \
  --cremad_dir /home/ubuntu/datasets/crema_d_videos \
  --output_dir models/wav2vec \
  --temp_dir temp_audio \
  --num_samples 10

echo
echo "=== File Diagnostics ==="
echo "Checking dataset directories:"
echo "RAVDESS Files (.mp4):"
find /home/ubuntu/datasets/ravdess_videos -name "*.mp4" | wc -l

echo "CREMA-D Files by extension:"
for ext in mp4 flv avi mov mkv wmv; do
  count=$(find /home/ubuntu/datasets/crema_d_videos -name "*.$ext" 2>/dev/null | wc -l)
  echo "  .$ext: $count files"
done

echo
echo "Extraction complete! Check the models/wav2vec directory for results."
echo "To process more samples, adjust the --num_samples parameter."
EOL
chmod +x run_enhanced_wav2vec_extraction.sh

# Upload the runner script
echo "Uploading runner script..."
scp -i "$KEY_PATH" run_enhanced_wav2vec_extraction.sh "$EC2_INSTANCE:$REMOTE_DIR/"

# Execute remotely
echo "Running enhanced wav2vec extraction on EC2..."
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "cd $REMOTE_DIR && chmod +x fixed_extract_wav2vec.py run_enhanced_wav2vec_extraction.sh && ./run_enhanced_wav2vec_extraction.sh"

echo
echo "=== Enhanced wav2vec Audio Extraction Complete ==="
echo "The wav2vec features have been extracted from video files on EC2."
echo
echo "To access the extracted features:"
echo "scp -i $KEY_PATH -r $EC2_INSTANCE:$REMOTE_DIR/models/wav2vec/ ./ec2_wav2vec_features/"
echo
echo "To process more files, connect to the EC2 instance and run the script with different parameters:"
echo "ssh -i $KEY_PATH $EC2_INSTANCE"
echo "cd $REMOTE_DIR"
echo "python fixed_extract_wav2vec.py --num_samples 100  # Adjust as needed"
