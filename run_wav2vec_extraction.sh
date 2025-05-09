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
