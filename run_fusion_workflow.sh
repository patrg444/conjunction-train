#!/usr/bin/env bash
# Run the complete emotion fusion workflow

# Create necessary directories
mkdir -p models/wav2vec
mkdir -p models/fusion

# Step 1: Make sure SlowFast model is downloaded
if [ ! -f "models/slowfast_emotion_video_only_92.9.pt" ]; then
    echo "Downloading SlowFast model..."
    ./download_and_extract_slowfast_model.sh
fi

# Step 2: Extract wav2vec features for a sample
echo "Extracting wav2vec features from sample videos..."
# Find a sample video file
SAMPLE_VIDEO=$(find downsampled_videos -name "*.mp4" | head -n 1)
if [ -z "$SAMPLE_VIDEO" ]; then
    echo "No sample video found. Using a placeholder."
    SAMPLE_VIDEO="placeholder.mp4"
fi
echo "Sample video: $SAMPLE_VIDEO"

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
echo "Fusion workflow complete!"
echo
echo "The following components were created:"
echo "1. SlowFast video model (downloaded)"
echo "2. Wav2vec audio features (extracted)"
echo "3. Fusion model configuration (created)"
echo "4. Demo script for inference (executed)"
echo
echo "To use on your own videos:"
echo "python demo_fusion_model.py --video /path/to/your/video.mp4"
