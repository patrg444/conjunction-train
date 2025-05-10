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
