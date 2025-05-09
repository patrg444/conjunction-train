#!/bin/bash
# Check the emotion coding in filenames for RAVDESS and CREMA-D datasets

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion/models/wav2vec"

echo "Examining wav2vec dataset emotion coding..."
echo "==========================================="

# Show sample filenames for CREMA-D
echo "CREMA-D Sample Filenames (first 10):"
ssh -i $KEY_PATH $EC2_HOST "ls -1 $REMOTE_DIR/cremad_*.npz | head -10"

# Show sample filenames for RAVDESS
echo -e "\nRAVDESS Sample Filenames (first 10):"
ssh -i $KEY_PATH $EC2_HOST "ls -1 $REMOTE_DIR/ravdess_*.npz | head -10"

# Extract the code that identifies emotion in filenames
echo -e "\nExamining actual code that parses emotions from filenames:"
ssh -i $KEY_PATH $EC2_HOST "grep -A20 'emotion_to_index' /home/ubuntu/audio_emotion/fixed_v5_script_continuous_indices.py"

# Look for emotion parsing code that extracts emotion codes from filenames
echo -e "\nCode that extracts emotion from filenames:"
ssh -i $KEY_PATH $EC2_HOST "grep -B10 -A30 'extract.*emotion' /home/ubuntu/audio_emotion/fixed_v5_script_continuous_indices.py"
ssh -i $KEY_PATH $EC2_HOST "grep -B10 -A30 'filename.*emotion' /home/ubuntu/audio_emotion/fixed_v5_script_continuous_indices.py"
ssh -i $KEY_PATH $EC2_HOST "grep -B10 -A30 'get.*emotion' /home/ubuntu/audio_emotion/fixed_v5_script_continuous_indices.py"
