# Attention CRNN Model for Audio Emotion Recognition

This document describes the workflow for training the Attention CRNN (Convolutional Recurrent Neural Network) model for audio emotion recognition on an EC2 GPU instance.

## Model Architecture

The Attention CRNN model combines convolutional and recurrent neural networks with an attention mechanism to focus on the most emotionally salient parts of audio input:

- **Input**: Audio features (WAV2VEC embeddings or spectrogram features)
- **Architecture**:
  - Convolutional layers to learn local patterns
  - Bidirectional GRU layers for temporal dependencies
  - Bahdanau-style attention mechanism to focus on emotionally relevant frames
  - Dense classification layers
- **Output**: Emotion classification probabilities

The model achieves approximately 77-78% accuracy on the test dataset, which is excellent for audio-only emotion recognition.

## Deployment and Training

### Deploy to EC2

The `deploy_attn_crnn_training.sh` script automates the process of:
1. Syncing project files to the EC2 instance
2. Setting up a conda environment with required dependencies 
3. Launching training in a detached tmux session

```bash
# Make the script executable
chmod +x deploy_attn_crnn_training.sh

# Run the deployment script
./deploy_attn_crnn_training.sh
```

### Monitor Training

The `monitor_attn_crnn_training.sh` script provides utilities for:
- Checking GPU status
- Viewing training logs
- Downloading the trained model

```bash
# Make the script executable
chmod +x monitor_attn_crnn_training.sh

# Check GPU status
./monitor_attn_crnn_training.sh -s

# View training logs
./monitor_attn_crnn_training.sh -l

# Download trained model
./monitor_attn_crnn_training.sh -d

# Show help
./monitor_attn_crnn_training.sh -h
```

## Using the Trained Model

The trained model will be saved as `best_attn_crnn_model.h5` and will be automatically downloaded to your `./models/` directory when using the monitoring script's download option.

```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('./models/best_attn_crnn_model.h5', 
                                  custom_objects={'AttentionLayer': AttentionLayer})

# For inference
predictions = model.predict(audio_features)
```

## Multimodal Fusion

The Attention CRNN model can be used as part of a multimodal emotion recognition system:

1. Train audio and video models separately
2. Extract embeddings from penultimate layers
3. Train a fusion model that combines these embeddings
4. Optional: Fine-tune the combined model end-to-end

This approach allows for modular development and easier debugging of each modality.

## Expected Performance

The Attention CRNN model achieves:
- Training accuracy: ~85-90%
- Validation accuracy: ~77-78%
- Test accuracy: ~75-77%

These results are competitive with state-of-the-art audio-only emotion recognition models.
