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
