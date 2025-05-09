#!/usr/bin/env bash
# Deploy and run the wav2vec extraction script on EC2

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "=== Deploying wav2vec Audio Extraction to EC2 ==="
echo "EC2 Instance: $EC2_INSTANCE"
echo "Remote Directory: $REMOTE_DIR"
echo

# Create remote directory if it doesn't exist
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "mkdir -p $REMOTE_DIR"

# Upload the script
echo "Uploading wav2vec extraction script..."
scp -i "$KEY_PATH" extract_audio_and_wav2vec_fusion.py "$EC2_INSTANCE:$REMOTE_DIR/"

# Create and upload a runner script
cat > run_wav2vec_extraction.sh << 'EOL'
#!/bin/bash
# Run the wav2vec extraction script on EC2

# Activate the PyTorch environment
source /opt/pytorch/bin/activate

# Install required packages
pip install moviepy

# Make directories
mkdir -p models/wav2vec
mkdir -p models/fusion

# Run the extraction script with a limited number of samples for testing
python extract_audio_and_wav2vec_fusion.py \
  --ravdess_dir /home/ubuntu/datasets/ravdess_videos \
  --cremad_dir /home/ubuntu/datasets/crema_d_videos \
  --output_dir models/wav2vec \
  --num_samples 10

echo
echo "Extraction complete! Check the models/wav2vec directory for results."
echo "To process more samples, adjust the --num_samples parameter."
EOL
chmod +x run_wav2vec_extraction.sh

# Upload the runner script
echo "Uploading runner script..."
scp -i "$KEY_PATH" run_wav2vec_extraction.sh "$EC2_INSTANCE:$REMOTE_DIR/"

# Execute remotely
echo "Running wav2vec extraction on EC2..."
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "cd $REMOTE_DIR && chmod +x extract_audio_and_wav2vec_fusion.py run_wav2vec_extraction.sh && ./run_wav2vec_extraction.sh"

echo
echo "=== wav2vec Audio Extraction Complete ==="
echo "The wav2vec features have been extracted from MP4 files on EC2."
echo
echo "To access the extracted features:"
echo "scp -i $KEY_PATH -r $EC2_INSTANCE:$REMOTE_DIR/models/wav2vec/ ./ec2_wav2vec_features/"
echo
echo "To process more files, connect to the EC2 instance and run the script with different parameters:"
echo "ssh -i $KEY_PATH $EC2_INSTANCE"
echo "cd $REMOTE_DIR"
echo "python extract_audio_and_wav2vec_fusion.py --num_samples 100  # Adjust as needed"
