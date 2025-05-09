# CNN-LSTM Model for Emotion Recognition

This project implements a CNN-LSTM architecture for audio emotion recognition using pre-extracted CNN features.

## Data Structure

The model uses CNN audio features from two datasets:
- RAVDESS dataset: Stored in `/home/ubuntu/emotion-recognition/data/ravdess_features_cnn_fixed/`
- CREMA-D dataset: Stored in `/home/ubuntu/emotion-recognition/data/crema_d_features_cnn_fixed/`

Total feature count: 8882 audio files (verified via file count).

## Model Architecture

The implementation explores two complementary approaches:
1. **BiLSTM Model**: Captures temporal patterns in variable-length audio sequences
   - Bidirectional LSTM layers for sequence processing
   - Time-distributed dense layers for feature extraction
   - Masking layer to handle variable-length inputs

2. **Global Pooling Model**: Captures overall feature distributions
   - Global average pooling for dimension reduction
   - Dense layers for feature abstraction
   - More robust to length variations

## Key Implementation Features

### Variable-Length Sequence Handling

The model properly handles variable-length input sequences through:
- Proper batch padding to maximum sequence length within each batch
- Masking to ignore padded regions during training
- Tracking of sequence lengths for efficient memory usage

```python
# Example of batch handling from the data generator
sequence_lengths = []
for features in batch_x:
    sequence_lengths.append(features.shape[0])

# Pad sequences to the maximum length in this batch
max_len = max(sequence_lengths)
padded_batch_x = []

for features in batch_x:
    # Create padded version of the features
    padded_features = np.zeros((max_len, feature_dim))
    # Copy actual features to the beginning
    padded_features[:features.shape[0], :] = features
    padded_batch_x.append(padded_features)
```

### Training Details

- Cross-entropy loss function
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting
- Model checkpointing for best validation accuracy
- Training proceeds with a 80/20 train/validation split

## Deployment

The model is deployed to AWS EC2 using:
- `deploy_fixed_cnn_lstm_model_to_ec2.sh`: Transfers code and launches training
- `monitor_ec2_cnn_lstm_training.sh`: Interactive monitoring of training progress

## Results

Models are saved to the `models/cnn_lstm_fixed_TIMESTAMP/` directory:
- `lstm_model.h5`: BiLSTM-based temporal model
- `pooling_model.h5`: Global pooling-based model

The final performance comparison shows which approach performs better on the validation set.

## Monitoring

Use the monitoring script for real-time tracking:
```bash
./monitor_ec2_cnn_lstm_training.sh
```

This provides:
- GPU usage statistics
- Live training log streaming
- Model performance summary
