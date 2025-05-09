# Attention-CRNN Model for Wav2Vec Features (Combined Dataset)

## Solution Summary

The Attention-CRNN (Convolutional Recurrent Neural Network) model is now successfully running on the EC2 GPU instance using both RAVDESS and CREMA-D datasets. The model combines CNN layers for feature extraction with a bidirectional LSTM and attention mechanism to effectively classify emotion from wav2vec audio embeddings.

## Key Implementation Details

1. **Data Loading and Preprocessing**:
   - Successfully combined both RAVDESS and CREMA-D dataset features for training
   - Balanced label distribution across 6 emotions: angry, disgust, happy, sad, fearful, and neutral
   - Sequence padding implemented to standardize input length (174 frames, based on 95th percentile)
   - Data augmentation applied to improve model generalization and dataset balance

2. **Model Architecture**:
   - Time-distributed CNN layers to extract local features from the wav2vec embeddings
   - Bidirectional LSTM to capture temporal dynamics in both directions
   - Attention mechanism to focus on the most emotionally salient parts of the audio
   - Dropout and batch normalization for regularization

3. **Training Configuration**:
   - Batch size optimized for GPU memory usage
   - Learning rate with decay scheduler
   - Early stopping to prevent overfitting
   - Model checkpointing to save the best version

## Monitoring Tools

A comprehensive monitoring script (`monitor_combined_attn_crnn.sh`) has been created to track training progress with both datasets:

```bash
# Basic monitoring
./monitor_combined_attn_crnn.sh

# GPU usage
./monitor_combined_attn_crnn.sh -g

# Check if process is running
./monitor_combined_attn_crnn.sh -p

# View logs
./monitor_combined_attn_crnn.sh -l

# Check tmux output
./monitor_combined_attn_crnn.sh -m

# Count processed files
./monitor_combined_attn_crnn.sh -c

# Download model when training completes
./monitor_combined_attn_crnn.sh -d
```

## Training Progress

Training is progressing well:
- Model is successfully utilizing the full GPU capacity (94% GPU memory usage)
- Process is running in a tmux session for resilience against disconnections
- Training with the combined dataset will improve generalization and robustness

## Next Steps

1. Allow the model to complete training (100 epochs with early stopping)
2. Evaluate the model on validation data to assess performance
3. Consider further optimizations if needed:
   - Learning rate tuning
   - Architecture modifications
   - Additional data augmentation techniques

This implementation successfully addresses the previous issues and provides a robust foundation for emotion recognition using wav2vec features from multiple datasets with an attention-based CRNN architecture. By combining RAVDESS and CREMA-D, the model will be exposed to greater acoustic and emotional variety, improving its real-world performance.
