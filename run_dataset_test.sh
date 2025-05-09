#!/usr/bin/env bash
# Run script for testing the dataset utilities on the EC2 instance

set -e  # Exit on any error

# Make the test script executable
chmod +x emotion_comparison/test_dataset.py

# Create processed data directory if it doesn't exist
mkdir -p processed_data

# Run the test script with EC2 paths
echo "Running dataset test script..."
python -m emotion_comparison.test_dataset \
    --ravdess_dir "./downsampled_videos/RAVDESS" \
    --cremad_dir "./downsampled_videos/CREMA-D-audio-complete" \
    --output_dir "./processed_data"

echo "Test completed. Check the processed_data directory for results."
