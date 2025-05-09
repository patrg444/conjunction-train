# Wav2Vec Audio Model Training: Numerical Stability Guide

This document explains the numerical stability improvements implemented in the fixed wav2vec audio emotion recognition training pipeline. These improvements address the NaN/Inf issues that can occur during training when using wav2vec embeddings.

## Root Causes of Numerical Instability

When training neural networks on wav2vec embeddings, several factors can lead to numerical instability:

1. **Extreme Feature Values**: Wav2vec features can sometimes have extreme values (very large positive or negative) that exceed the normal range of neural network activations.
2. **Batch Normalization Issues**: Traditional batch normalization computes moving averages which can be affected by extreme values in early batches.
3. **Zero-only Padding Domination**: Sequence padding can lead to skewed statistics when a large portion of the input consists of zeros.
4. **Gradient Explosions**: LSTM cells are particularly vulnerable to this when processing long sequences.
5. **Precision Limitations**: Default mixed-precision training can introduce instability in some operations.

## Implemented Solutions

### 1. Robust Feature Loading (`wav2vec_fixed_loader.py`)

The feature loading process has been completely revised to include multiple safeguards:

```python
def improved_load_features(file_path, mean, std, clip_value=5.0):
    # Load with safe type
    features = np.load(file_path).astype(np.float32)
    
    # Replace NaN/Inf with zeros
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features)
    
    # Clip extreme values to a reasonable range
    features = np.clip(features, -clip_value, clip_value)
    
    # Use epsilon in normalization to avoid division by zero
    epsilon = 1e-7
    std_safe = np.maximum(std, epsilon)
    normalized = (features - mean) / std_safe
    
    return normalized
```

### 2. Architecture Modifications (`train_wav2vec_audio_only_fixed.py`)

The model architecture has been redesigned with stability in mind:

1. **Layer Normalization Instead of Batch Normalization**:
   ```python
   # Use LayerNormalization which has no running statistics
   x = LayerNormalization(epsilon=1e-6)(x)
   ```

2. **Constrained Weights with MaxNorm**:
   ```python
   # Apply constraints to LSTM weights
   kernel_constraint=MaxNorm(3.0),
   recurrent_constraint=MaxNorm(3.0)
   ```

3. **Stabilized Attention Mechanism**:
   ```python
   # Numeric stability in softmax calculation
   scores_max = tf.reduce_max(scores, axis=1, keepdims=True)
   scores_shifted = scores - scores_max  # Subtract max for stability
   attention_weights = tf.nn.softmax(scores_shifted, axis=1)
   ```

4. **Logit Clipping Before Softmax**:
   ```python
   x = Lambda(lambda z: tf.clip_by_value(z, -15.0, 15.0), name='logit_clipping')(x)
   ```

### 3. Training Process Improvements

1. **Explicit Float32 Precision**:
   ```python
   # Explicitly set to float32 for numerical stability
   tf.keras.mixed_precision.set_global_policy("float32")
   ```

2. **SGD with Gradient Clipping**:
   ```python
   optimizer = SGD(
       learning_rate=learning_rate,
       momentum=0.9,
       nesterov=True,
       clipnorm=1.0,  # Clip gradients by norm
       clipvalue=0.5  # Also clip by value
   )
   ```

3. **Gradual Learning Rate Warm-up**:
   ```python
   # Start with 100x smaller learning rate
   learning_rate = args.lr / 100
   
   # Warm-up over 10 epochs
   if epoch < self.warmup_epochs:
       learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
   ```

4. **Label Smoothing in Loss Function**:
   ```python
   model.compile(
       optimizer=optimizer,
       loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
       metrics=['accuracy']
   )
   ```

5. **NaN Detection and Handling**:
   ```python
   class NaNLossDetector(Callback):
       def on_batch_end(self, batch, logs=None):
           logs = logs or {}
           loss = logs.get('loss')
           if loss is not None and (math.isnan(loss) or math.isinf(loss)):
               print(f'NaN/Inf loss detected: {loss} at batch {batch}, stopping training.')
               self.model.stop_training = True
   ```

6. **Operation Determinism**:
   ```python
   # Enable operation determinism for reproducibility
   tf.config.experimental.enable_op_determinism()
   ```

7. **Debug Mode with Numeric Checking**:
   ```python
   if args.debug:
       tf.debugging.enable_check_numerics()
   ```

## Usage Guide

### Setup and Deployment

1. **Prepare Scripts**: Ensure all the necessary files are in place:
   - `scripts/wav2vec_fixed_loader.py`: Robust feature loading module
   - `scripts/train_wav2vec_audio_only_fixed.py`: Numerically stable training script
   - `run_wav2vec_audio_only_fixed.sh`: Runner script with optimized parameters

2. **Deploy to EC2**:
   ```bash
   ./deploy_fixed_wav2vec_training.sh
   ```
   This uploads all scripts, makes them executable, and sets up TensorBoard.

### Training

1. **Monitoring Training Progress**:
   ```bash
   ./monitor_wav2vec_fixed_training.sh
   ```
   This provides real-time insights into:
   - GPU utilization
   - Training metrics
   - Potential NaN/Inf values
   - Learning rate adjustments

2. **TensorBoard Visualization**:
   ```bash
   ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@54.162.134.77
   ```
   Then open http://localhost:6006 in your browser.

### Post-Training

1. **Download Trained Model**:
   ```bash
   ./download_wav2vec_fixed_model.sh
   ```
   This downloads:
   - Model weights
   - Training history
   - Log files
   - TensorBoard logs

2. **Training Curve Analysis**:
   The download script automatically generates training curves for analysis.

## Progressive Training Strategy

For the most stable training process, follow this progression:

1. **Start Small**: Begin with a small batch size (8) for the first 30 epochs.
2. **Verify Stability**: Use the monitoring script to confirm no NaN/Inf values appear.
3. **Scale Up Gradually**: If training is stable, restart with a larger batch size (32).
4. **Extended Learning**: Enable up to 200 epochs with early stopping for best results.

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss becomes NaN early | Learning rate too high | Already addressed with warm-up and lower initial LR |
| Gradients contain NaN | Extreme feature values | Addressed with feature clipping and gradient clipping |
| Model diverges after initial progress | Saturation in activation functions | Fixed with logit clipping and layer normalization |
| Memory issues with large batches | Sequence padding inefficiency | Use dynamic sequence lengths with smaller batches |

## Performance Expectations

With these stability improvements, you should expect:

- **No NaN/Inf Issues**: Training should complete without numerical errors
- **Stable Convergence**: Loss should decrease smoothly without spikes
- **Comparable Accuracy**: Final model should achieve similar or better accuracy (~70-75% on validation data)
