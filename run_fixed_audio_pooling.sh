#!/bin/bash
# Run the fixed audio-pooling LSTM training script with command-line arguments
# Usage: ./run_fixed_audio_pooling.sh [epochs] [batch_size] [seq_len]
# Default values: 1 epoch for testing, 24 batch size, 45 seq_len

EPOCHS=${1:-1}      # Default to 1 epoch for testing
BATCH_SIZE=${2:-24} # Default batch size
SEQ_LEN=${3:-45}    # Default sequence length (15 FPS × 3 s window)

echo "Running fixed audio-pooling LSTM training with:"
echo "- EPOCHS: $EPOCHS"
echo "- BATCH_SIZE: $BATCH_SIZE"
echo "- MAX_SEQ_LEN: $SEQ_LEN"
python scripts/train_audio_pooling_lstm_fixed.py --epochs $EPOCHS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "- Model saved as model_best.h5"
    echo "- Metadata saved as model_info.json"
    echo "- Normalization stats saved to audio_normalization_stats.pkl and video_normalization_stats.pkl"
    
    # Optionally copy to inference location if needed
    # cp model_best.h5 demo_app/
    # cp model_info.json demo_app/
    # cp audio_normalization_stats.pkl video_normalization_stats.pkl demo_app/
else
    echo "Training failed with exit code $?"
fi
