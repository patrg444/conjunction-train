# LSTM Attention Model with Synchronized Augmentation

This implementation enhances the LSTM Attention model by adding **synchronized audio-visual augmentation** that preserves the temporal alignment between modalities.

## Key Innovations

This model builds on the success of the LSTM attention model without augmentation by adding:

1. **Synchronized Time Warping** - Applies consistent time stretching/compression to both audio and video features while maintaining their temporal relationship
2. **Correlated Noise Addition** - Adds appropriately scaled noise to both modalities in a way that preserves their natural relationship
3. **Data Amplification** - Increases the effective training set size without breaking critical temporal alignment patterns

## Technical Details

### Data Generator Architecture

The augmentation is implemented through a custom data generator (`SynchronizedAugmentationDataGenerator`) that:

- Preserves the original samples intact
- Creates augmented variations with strictly correlated transformations
- Maintains batch-wise dynamic padding for efficient processing of variable-length sequences
- Controls augmentation intensity through configurable parameters

### Augmentation Techniques

#### 1. Synchronized Time Warping

```python
# Apply the same stretch factor to both modalities
stretch_factor = np.random.uniform(0.9, 1.1)  # 90-110% of original length

# Calculate new lengths maintaining the ratio
new_audio_len = max(5, int(audio_len * stretch_factor))
new_video_len = max(5, int(video_len * stretch_factor))

# Resample both sequences with proportionally consistent indices
audio_indices = np.linspace(0, audio_len-1, new_audio_len)
video_indices = np.linspace(0, video_len-1, new_video_len)
```

#### 2. Correlated Noise Addition

```python
# Create correlated noise levels
audio_noise_level = np.random.uniform(0.001, 0.01)

# Scale video noise level based on relative standard deviations
video_noise_level = audio_noise_level * (video_std / audio_std)

# Apply noise that maintains the natural relationship between modalities
audio_result += audio_noise
video_result += video_noise
```

## Expected Performance

Building on the strong foundation of the LSTM attention model, we expect the synchronized augmentation approach to provide:

- **Improved Generalization**: By creating more training variations without breaking the temporal alignment crucial for emotion recognition
- **Enhanced Robustness**: Against minor variations in audio or visual quality
- **Similar or Better Accuracy**: Compared to the no-augmentation model (82.9+%), potentially reaching 85-88% validation accuracy

## Usage

### Training the Model

To train the model on AWS EC2:

```bash
./aws-setup/deploy_lstm_attention_sync_aug.sh
```

This script:
1. Creates an EC2 instance with optimal configuration for training
2. Uploads the necessary code and data to the instance
3. Sets up the Python environment with required dependencies
4. Starts the training process in the background

### Monitoring Training Progress

```bash
./aws-setup/live_stream_training.sh
```

### Downloading Results

```bash
./aws-setup/download_lstm_attention_sync_aug_results.sh
```

### Local Testing 

If you want to test locally before deploying to EC2:

```bash
python3 scripts/train_branched_attention_sync_aug.py
```

## Technical Considerations

The key innovation in this approach is that **augmentation is applied with strict synchronization between modalities**. This is crucial for multimodal emotion recognition because:

1. Emotions are expressed through coordinated facial expressions and vocal patterns
2. The relationship between audio and visual features is temporally sensitive
3. Independent augmentation of modalities can create implausible combinations that harm model training

This approach is expected to provide the benefits of data augmentation without introducing the temporal misalignment problems that would occur with independent augmentation of each modality.

## Implementation Details

- **Augmentation Factor**: The dataset is effectively expanded by a factor of 2x through augmentation
- **Augmentation Probability**: Each eligible sample has an 80% chance of being augmented
- **Model Architecture**: Maintains the same attention-based LSTM architecture as the no-augmentation model
- **Focal Loss**: Retains the focal loss implementation for handling class imbalance

## System Requirements

- TensorFlow 2.6+
- NumPy 1.19.5+
- Python 3.6+
- For AWS deployment: C5.24xlarge instance (recommended)
