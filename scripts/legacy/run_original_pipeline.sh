#!/bin/bash
# Wrapper script to run the original emotion recognition pipeline
# using the legacy environment with TensorFlow 1.15

# Activate the legacy environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate emotion_recognition_legacy

# Run the original pipeline script
python scripts/realtime_emotion_recognition.py "$@"

# Deactivate the environment when done
conda deactivate
