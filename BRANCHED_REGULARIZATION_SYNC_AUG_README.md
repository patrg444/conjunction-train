# Combined Model with Four Key Improvements for Emotion Recognition

This implementation combines four successful model improvements to achieve higher validation accuracy:

1. **L2 Regularization** - Adds weight penalties to dense layers to improve generalization
2. **Synchronized Data Augmentation** - Increases the effective training set size while preserving temporal alignment between audio and visual modalities
3. **Optimized Learning Rate** - Uses a higher initial learning rate (0.001 instead of 0.0005) for better convergence
4. **Temporal Convolutional Network (TCN)** - Replaces Bidirectional LSTM in the video branch with TCN for better temporal modeling

## Expected Performance

The baseline branched model without these improvements achieves approximately 84.14% validation accuracy on the combined RAVDESS and CREMA-D datasets. From our experiments:

- **Branched Regularization only**: ~84.8% validation accuracy (+0.7%)
- **Synchronized Augmentation only**: ~84.8% validation accuracy (+0.7%) 
- **Branched Optimizer only**: ~84.36% validation accuracy (+0.22%)
- **TCN for Video Branch**: Expected improvement due to better temporal modeling
- **Combined approach with TCN**: Expected 86%+ validation accuracy

## Technical Details

### 1. L2 Regularization

L2 regularization with strength 0.001 is applied to all dense layers, including:
- Audio branch dense layer
- Video branch dense layer 
- All merged layers
- Output layer

This helps prevent overfitting by penalizing large weights and encouraging the model to learn more generalizable features.

### 2. Synchronized Data Augmentation

The synchronized augmentation preserves the temporal relationship between audio and visual signals through:

1. **Synchronized Time Warping** - Applying the same time stretching factor to both modalities
2. **Correlated Noise Addition** - Adding appropriately scaled noise to maintain cross-modal relationships
3. **Dynamic Batch Padding** - Maintaining efficient processing of variable-length sequences

With an augmentation factor of 2.0 and 80% augmentation probability, the effective training dataset size is roughly doubled.

## Implementation

### Model Architecture

The model architecture has been enhanced with TCN for the video branch:

```
Audio Input → Masking → Conv1D → BatchNorm → MaxPooling → Conv1D → BatchNorm → MaxPooling → BiLSTM → Dropout → BiLSTM → Dense+L2 → Dropout

Video Input → Masking → TCN(dilation=1) → TCN(dilation=2) → TCN(dilation=4) → TCN(dilation=8) → GlobalAveragePooling1D → Dense+L2 → Dropout

Concatenate[Audio Branch, Video Branch] → Dense+L2 → BatchNorm → Dropout → Dense+L2 → BatchNorm → Dropout → Dense+L2 (Output)
```

The TCN implementation includes:
- Residual connections between blocks
- Increasing dilation rates (1, 2, 4, 8) for wider temporal receptive field
- Causal convolutions to prevent information leakage
- L2 regularization applied to convolutional layers

### Key Code Additions

L2 regularization is applied to all dense layers:

```python
audio_x = Dense(128, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(audio_x)
video_x = Dense(256, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(video_x)
merged = Dense(256, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(merged)
# etc.
```

TCN is implemented using residual blocks with dilated convolutions:

```python
def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2, l2_reg=0.001):
    # First dilated convolution
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',  # Important for causality - only looking at past
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Second dilated convolution
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    
    # Residual connection
    if x.shape[-1] != filters:
        # If dimensions don't match, use 1x1 conv to adapt dimensions
        x = Conv1D(filters=filters, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))(x)
    
    # Add residual connection
    result = Add()([conv2, x])
    return Activation('relu')(result)
```

Synchronized augmentation is achieved through the SynchronizedAugmentationDataGenerator:

```python
train_generator = SynchronizedAugmentationDataGenerator(
    train_video, train_audio, train_labels,
    batch_size=BATCH_SIZE, 
    shuffle=True,
    augmentation_factor=AUGMENTATION_FACTOR,
    augmentation_probability=0.8
)
```

### 3. Optimized Learning Rate

The model uses a higher learning rate (0.001) compared to the baseline model (0.0005), which helps:
- Faster initial convergence during training
- Better escape from local minima
- More thorough exploration of the parameter space

This is combined with the existing learning rate scheduler (ReduceLROnPlateau) which reduces the learning rate when progress plateaus, providing both the benefits of a higher initial learning rate and the stability of a lower rate later in training.

## Theoretical Basis

The combination of these four approaches is synergistic:

1. **Regularization + Augmentation**: L2 regularization prevents the model from overfitting to the augmented data
2. **Data Diversity + Generalization**: The diverse examples from augmentation help the model learn better representations, while regularization ensures they generalize
3. **Robustness Improvement**: The model becomes robust to minor variations in both audio and visual inputs
4. **TCN Advantages**: The TCN architecture offers several benefits over BiLSTM for the video branch:
   - Parallelizable computation for faster training
   - Ability to capture long-range dependencies through dilated convolutions
   - More stable gradient flow during backpropagation
   - Explicit control over the receptive field size
   - Better handling of varied sequence lengths

## Usage

### Training the Model on AWS EC2

```bash
./aws-setup/deploy_branched_regularization_sync_aug.sh
```

Note: The model outputs will be saved to the `models/branched_regularization_sync_aug_tcn` directory.

This script:
1. Creates an EC2 instance with optimal configuration for training
2. Uploads the necessary code and data to the instance
3. Sets up the Python environment with required dependencies
4. Starts the training process in the background

### Monitoring Training Progress

```bash
cd aws-setup && ./monitor_training.sh training_branched_regularization_sync_aug.log
```

## System Requirements

- TensorFlow 2.6+
- NumPy 1.19.5+
- Python 3.6+
- For AWS deployment: C5.24xlarge instance (recommended)
