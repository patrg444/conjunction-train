# SlowFast-R50 Emotion Recognition Training

This implementation aims to improve the visual emotion recognition accuracy from ~60% to ~70% using the SlowFast-R50 backbone with advanced augmentation techniques.

## What's Included

- **SlowFast-R50 Backbone**: A more powerful 3D CNN architecture than the previous R3D-18.
- **Enhanced Augmentation Pipeline**: Multiple augmentation techniques to improve generalization.
- **Multi-clip Sampling**: Training on different temporal segments of each video.
- **AdamW + Label Smoothing**: Modern optimization techniques for better convergence.
- **Test-time Ensembling**: Averaging predictions from multiple clips at inference time.

## Key Improvements

1. **Higher Capacity Backbone**: SlowFast-R50 has a two-pathway structure that processes video at both slow and fast frame rates, capturing better temporal information.

2. **Advanced Augmentations**:
   - **Spatial**: Random affine transformations, cutout patches
   - **Temporal**: Time reversal, frame dropping
   - **Color**: Enhanced color jitter 

3. **Training Improvements**:
   - Label smoothing to reduce overfitting
   - AdamW optimizer with weight decay
   - One-cycle learning rate schedule

4. **Multiple Clips**: Sampling 2+ clips per video during training and 5 clips during validation improves temporal coverage.

5. **Squeeze-and-Excitation Blocks**: Added to the backbone to enhance feature quality through channel-wise attention.

## Usage

### Deploy and Start Training

```bash
# Deploy and start training on EC2
./deploy_slowfast_training.sh

# Monitor training progress
./monitor_slowfast_training.sh
```

### Download Trained Model

After training is complete or you want to retrieve the latest checkpoint:

```bash
# Download the best model
./download_slowfast_model.sh
```

## Expected Results

- **Previous R3D-18**: ~60% validation accuracy
- **SlowFast-R50**: ~68-72% validation accuracy (expected)

With properly tuned audio features (~75% accuracy), the fusion model could achieve:
- **Late Fusion**: ~78-82% accuracy
- **Attention-based Fusion**: ~82-85% accuracy

## Technical Details

The implementation uses:
- Face cropping at 112×112 resolution
- 48 frames per clip
- Batch size of 6 (due to larger model)
- Mixed precision training (FP16)
- Cosine learning rate schedule
- Early stopping with patience of 12 epochs

## Next Steps

1. Complete training of the visual model
2. Train the audio branch with Wav2Vec 2.0 or log-mel LSTM
3. Create the joint embedding dataset
4. Implement cross-modal attention fusion
5. Train the fusion model
