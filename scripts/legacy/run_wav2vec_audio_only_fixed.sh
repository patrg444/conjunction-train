#!/bin/bash
# Run the fixed wav2vec training with aggressive numerical stability measures

# Set environment variable to increase TensorFlow logging (shows operation failures)
export TF_CPP_MIN_LOG_LEVEL=0

# Set TF logging to be even more verbose about numerical issues
export TF_ENABLE_ONEDNN_OPTS=0

# Enable core dumps in case it crashes
ulimit -c unlimited

# Create timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="wav2vec_fixed_training_${TIMESTAMP}.log"

echo "Starting fixed wav2vec training at $(date)"
echo "Training logs will be saved to ${LOG_FILE}"

# Use a small batch size initially to test for stability
BATCH_SIZE=8

# Run with debug mode to catch NaN issues early
python -m scripts.train_wav2vec_audio_only_fixed \
  --features_dir /home/ubuntu/audio_emotion \
  --mean_path audio_mean.npy \
  --std_path audio_std.npy \
  --epochs 200 \
  --batch_size ${BATCH_SIZE} \
  --lr 1e-4 \
  --log_dir logs \
  --checkpoint_dir checkpoints \
  --debug \
  --clip_value 5.0 \
  2>&1 | tee ${LOG_FILE}

echo "Training completed at $(date)"
echo "Check ${LOG_FILE} for details"

# After 30 epochs with no NaNs, you can restart with a larger batch size:
# BATCH_SIZE=32
