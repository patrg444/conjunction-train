# Optimized Model with Balanced Architecture and Enhanced Regularization

This implementation combines several carefully calibrated improvements to achieve higher validation accuracy:

1. **Enhanced L2 Regularization** - Increased from 0.001 to 0.0015 for better generalization
2. **Synchronized Data Augmentation** - Maintains temporal alignment between audio/video modalities
3. **Focused TCN (Temporal Convolutional Network)** - Optimized structure with 4 blocks instead of 6
4. **Balanced Model Capacity** - Moderate parameter increase with optimal proportions between branches
5. **Increased Learning Rate** - Raised from 0.0001 to 0.0005 for better parameter space exploration

## Optimization Approach

After observing 83.8% accuracy from the initial large model (below target), we identified several issues:

1. **Learning Rate Limitations**: The very low learning rate (0.0001) was likely causing:
   - Slow convergence and difficult escape from local minima
   - Inability to explore parameter space effectively

2. **Architectural Imbalance**: The aggressive scaling of all components created:
   - Potentially suboptimal proportion between audio and video branches
   - Too many parameters relative to the regularization strength

3. **Excessive Temporal Context**: The 6-block TCN with dilations up to 32 created:
   - Too wide a receptive field (63 timesteps)
   - Potential inclusion of irrelevant temporal information

## Key Architecture Refinements

The optimized model uses a more balanced scaling approach:

### Audio Branch
- **Convolutional Layers**: Moderate increase from [64, 128] to [80, 160] filters
- **LSTM Units**: Moderate increase from [128, 64] to [160, 80] units

### Video Branch with TCN
- **TCN Filters**: Moderate increase from 128 to 160 filters per block
- **TCN Blocks**: Reverted to 4 blocks for more focused temporal modeling
- **Dilation Rates**: Maintained effective [1,2,4,8] for 15-timestep receptive field
- **Dropout Rate**: Increased from 0.2 to 0.25 for better regularization

### Merged Branch
- **Dense Layers**: Moderate increase from [256, 128] to [320, 160] units

## Expected Performance

With these optimizations, we expect improved performance due to:

1. **Better Optimization Dynamics**: Higher learning rate allows more effective parameter search
2. **Enhanced Regularization**: Increased L2 strength and dropout better control the model capacity
3. **More Focused Temporal Modeling**: Optimized TCN structure focuses on relevant temporal patterns
4. **Balanced Architecture**: Better proportions between branches maintains cross-modal information flow

Our baseline branched model achieved around 84.14% validation accuracy, and we now target validation accuracy of 86-87% with these optimized parameters.

## Deployment and Monitoring

### Training the Optimized Model on AWS EC2

```bash
./aws-setup/deploy_branched_regularization_sync_aug_tcn_large.sh
```

### Monitoring Training Progress

```bash
./tcn_model_tracking.sh
```
Then select option "0" to switch to the optimized model, and "1" to monitor live logs.

## Implementation Details

The optimized model contains approximately 1.8x more parameters than the regular version, compared to 2.5x in the initial large model. This represents a more balanced increase in capacity while maintaining effective regularization.

```python
# Parameters for optimized model capacity
AUDIO_CONV_FILTERS = [80, 160]  # Moderate increase from [64, 128]
AUDIO_LSTM_UNITS = [160, 80]    # Moderate increase from [128, 64]
VIDEO_TCN_FILTERS = 160         # Moderate increase from 128
VIDEO_TCN_BLOCKS = 4            # Back to 4 blocks for focused temporal modeling
MERGED_DENSE_UNITS = [320, 160] # Balanced increase from [256, 128]
```

The learning rate scheduling has also been optimized for better convergence:
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,         # Less aggressive reduction (was 0.5)
    patience=4,         # More frequent adjustments (was 5)
    min_lr=5e-6,
    verbose=1
)
```

These changes aim to find the optimal balance between model capacity, regularization, and optimization dynamics.

## System Requirements

- TensorFlow 2.6+
- NumPy 1.19.5+
- Python 3.6+
- Recommended: GPU with 16GB+ VRAM or C5.24xlarge instance on AWS
