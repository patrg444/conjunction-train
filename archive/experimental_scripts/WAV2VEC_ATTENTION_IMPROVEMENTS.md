# WAV2VEC Emotion Recognition - Attention Model Improvements

## Problem Statement

The previous WAV2VEC emotion recognition model (v8/final_working_fix) was experiencing significant overfitting, reaching ~85% training accuracy but only ~48% validation accuracy before early stopping. This indicates the model was memorizing the training data rather than learning generalizable patterns.

## Key Improvements in v9 Attention Model

### 1. Attention Mechanism
- **Self-Attention Layer**: Replaced the second Bidirectional LSTM with a temporal attention mechanism that focuses on the most informative frames in the sequence
- **Global Pooling**: Added GlobalAveragePooling1D to convert attention output to a fixed-length representation
- **Why it matters**: Different emotions have salient markers at different points in audio; attention helps the model focus on these important segments rather than treating all frames equally

### 2. Enhanced Regularization
- **Layer Normalization**: Added after the first LSTM layer to stabilize training and improve gradient flow
- **Recurrent Dropout**: Added recurrent_dropout=0.25 to the LSTM to prevent overfitting in the recurrent connections
- **L2 Regularization**: Applied kernel_regularizer=l2(1e-4) to dense layers to penalize large weights
- **Adjusted Dropout Rates**: Increased to 0.4 after the first dense layer and 0.3 after the second (previously 0.5 for both)
- **Masking**: Added mask_zero=True to properly handle variable-length sequences

### 3. Optimized Learning Rate Schedule
- **Improved Warm-up**: Extended from 5 to 8 epochs to allow model to reach a more stable region before applying full learning rate
- **Lower Max LR**: Reduced from 1e-3 to 3e-4 to prevent overshooting optimal minima
- **Stronger LR Reduction**: Factor 0.3 (vs 0.5 previously) for more aggressive learning rate reduction when plateaus are reached
- **Cosine Decay**: Added cosine decay after 20 epochs to gently decrease learning rate for fine-tuning
- **Earlier Patience**: Reduced from 7 to 5 epochs for faster response to plateaus

### 4. Data Pipeline Enhancements
- **Train/Val Split Before Normalization**: Fixed data leakage by computing normalization statistics only on training data
- **Increased Sequence Length**: Using 98th percentile (vs 95th) to preserve more temporal information in longer utterances
- **Separate Normalization Files**: Using audio_mean_v9.npy and audio_std_v9.npy to avoid mixing with previous model statistics

### 5. Improved Monitoring and Callbacks
- **Monitor val_accuracy**: Both early stopping and checkpointing now track validation accuracy for consistency
- **Longer Early Stopping Patience**: Increased to 12 epochs (from original early stopping of 15 but LR reduction of 7)
- **Added TensorBoard Logging**: For better visualization of training progress

## Expected Performance Improvements

Based on the changes implemented, we expect:

1. **Reduced Overfitting**: The train-validation accuracy gap should be significantly smaller
2. **Higher Validation Accuracy**: Targeting 60-65% validation accuracy (vs ~45-48% previously)
3. **Better Generalization**: Model should perform more consistently across different emotion categories
4. **More Stable Training**: Learning curves should be smoother with fewer oscillations

## Usage Instructions

### Running a Smoke Test
```bash
./run_smoke_test.sh
```

### Deploying the Model to EC2
```bash
./deploy_v9_fix.sh
```

### Monitoring Training Progress
```bash
./monitor_v9_attention.sh
```

### Downloading the Trained Model
```bash
./download_v9_model.sh
```

## Technical Implementation Details

The core improvement is the addition of the attention mechanism:

```python
# Bidirectional LSTM with recurrent dropout for regularization
x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25))(masked_input)

# Add Layer Normalization for more stable training
x = LayerNormalization()(x)

# Self-attention mechanism to focus on important frames
context_vector = Attention(use_scale=True)([x, x])

# Global pooling to convert attention output to fixed-length representation
x = GlobalAveragePooling1D()(context_vector)
```

The learning rate schedule combines three phases:
1. Linear warm-up from 5e-6 to 3e-4 over 8 epochs
2. ReduceLROnPlateau with factor 0.3, patience 5 epochs
3. Cosine decay after epoch 20

This approach should find a better balance between exploration and exploitation during training.
