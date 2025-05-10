#!/bin/bash
# Check what emotion files exist in the wav2vec directory on the server

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion/models/wav2vec"

echo "Checking wav2vec dataset on remote server..."
echo "============================================="

# Check total count of files
echo "Total files in the directory:"
ssh -i $KEY_PATH $EC2_HOST "ls -1 $REMOTE_DIR/*.npz 2>/dev/null | wc -l"

# Check emotion distribution in filenames
echo -e "\nEmotion distribution in filenames:"
for emotion in neutral calm happy sad angry fear disgust surprise; do
    count=$(ssh -i $KEY_PATH $EC2_HOST "ls -1 $REMOTE_DIR/*_${emotion}_*.npz 2>/dev/null | wc -l")
    echo "  ${emotion}: ${count} files"
done

# Check pattern of filenames
echo -e "\nSample filenames (first 5 of each emotion if available):"
for emotion in neutral calm happy sad angry fear disgust surprise; do
    echo "  ${emotion}:"
    ssh -i $KEY_PATH $EC2_HOST "ls -1 $REMOTE_DIR/*_${emotion}_*.npz 2>/dev/null | head -5"
done

# Check different file naming patterns
echo -e "\nChecking alternate filename patterns..."
for position in 1 2 3 4 5; do
    echo "  Files with emotion in position $position of filename (split by '_'):"
    ssh -i $KEY_PATH $EC2_HOST "for f in $REMOTE_DIR/*.npz; do echo \$(basename \$f) | cut -d'_' -f$position; done | sort | uniq -c | sort -nr | head -10"
done

echo -e "\nExamining sample file structure:"
ssh -i $KEY_PATH $EC2_HOST "find $REMOTE_DIR -type f -name '*.npz' | head -1 | xargs python3 -c 'import sys, numpy as np; f=np.load(sys.argv[1]); print(\"Keys in NPZ file:\", list(f.keys())); print(\"Shapes:\", {k: f[k].shape for k in f.keys()})'"
