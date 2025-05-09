# G5.2xlarge GPU Training Configuration

This document outlines the complete process for training the 100-epoch audio-visual emotion recognition model on AWS G5.2xlarge GPU instance.

## Current Status

Training is currently running on the AWS G5.2xlarge instance:
- Public IP: 18.208.166.91
- Start time: 2025-04-19, approx. 13:25 PDT
- Expected run time: ~8-10 hours for 100 epochs
- Training script: `train_audio_pooling_lstm_with_laughter.py`

## Issues and Fixes Applied

We encountered and fixed several issues during the deployment:

1. **SSH Connectivity**:
   - Created new SSH key (gpu-key.pem) and properly configured permissions
   - Successfully established connection to G5.2xlarge instance

2. **Feature Normalization Issues**:
   - Added missing `load_normalization_stats` function to feature_normalizer.py
   - Fixed by creating a compatibility wrapper that calls the existing function

3. **Data Generator Issues**:
   - Original training script failed due to incompatible parameters in AudioPoolingDataGenerator
   - Fixed the AudioPoolingDataGenerator implementation to properly handle initialization parameters
   - Added support for laughter detection through an additional output branch

4. **Script Deployment Issues**:
   - Fixed parameter substitution in shell script by directly inserting variables
   - Added verification steps to confirm successful upload and execution

5. **Monitoring Enhancements**:
   - Set up TensorBoard for real-time training visualization
   - Created laughter manifest placeholder to enable multi-task learning
   - Implemented periodic model checkpointing for backup and progress tracking

## Available Scripts

| Script | Purpose |
|--------|---------|
| `deploy_fixed_gpu_training_v2.sh` | Deploys and starts the training on AWS G5 |
| `enhanced_monitor_g5.sh` | Advanced monitoring showing GPU usage, process status, etc. |
| `setup_enhanced_monitoring_fixed.sh` | Sets up TensorBoard and additional monitoring tools |
| `setup_tensorboard_tunnel.sh` | Creates persistent SSH tunnel for TensorBoard visualization |
| `download_g5_fixed_model_20250419_132500.sh` | Downloads the trained model after completion |

## Workflow Instructions

### 1. Monitor Training Progress

There are multiple ways to monitor the training progress:

#### Command-line Monitoring:
```bash
./enhanced_monitor_g5.sh
```

This shows:
- Training process status (running/stopped)
- Dataset presence and sizes
- Current GPU utilization and memory usage
- Recent log entries
- Estimated completion time based on current progress

#### TensorBoard Visualization:
```bash
./setup_tensorboard_tunnel.sh
```

This will:
- Create a persistent SSH tunnel to the EC2 instance
- Open TensorBoard in your browser automatically
- Show real-time training metrics, model graphs, and distribution statistics

### 2. Download Model After Completion

```bash
./download_g5_fixed_model_20250419_132500.sh
```

This will:
- Check if training is complete
- If complete, download model files to `models/g5_fixed_20250419_132500/`
- If not complete, show current status and exit

## Model Details

The model being trained is an audio-pooling LSTM with the following configuration:
- Unified audio-video processing architecture
- Audio features pooled to match video frame rate (15 FPS)
- LSTM units: [128, 64] for processing audio and video features
- Dense units: [256, 128] for final classification
- L2 regularization: 0.002
- Max norm constraint: 3.0
- Learning rate: 0.0006 with warm-up and cosine decay
- Sequence length: 45 frames (3 seconds at 15 FPS)
- Batch size: 256 (GPU optimized)
- Dual-task learning: emotion classification and laughter detection

## Validation

After the model is trained and downloaded, we will validate it by:
1. Running inference on test data to confirm accuracy
2. Testing with the real-time demo to verify real-world performance
3. Comparing metrics with previous models using the existing validation framework

## Troubleshooting

If the training process fails or stalls:
1. Check the EC2 console to verify instance is still running
2. Use `./enhanced_monitor_g5.sh` to see latest logs and process status
3. If needed, SSH directly to the instance to inspect detailed logs:
   ```
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91
   ```
4. If TensorBoard is not working, verify it's running with:
   ```
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "pgrep -f tensorboard"
   ```
5. You can restart the training if necessary with:
   ```
   ./deploy_fixed_gpu_training_v2.sh
   ```

## Data Concerns

Current training is running with:
- Missing or inaccessible feature archives (RAVDESS/CREMA-D)
- Placeholder laughter manifest instead of real data
- Dummy data generator fallback when no real data is found

This is acceptable for testing the pipeline, but before a production run:
1. Verify data presence on EC2: `du -sh ravdess_features_facenet/ crema_d_features_facenet/`
2. Upload proper laughter manifest: `datasets/manifests/laughter_v1.csv`
3. Ensure normalization stats are in place: `models/dynamic_padding_no_leakage/*.pkl`

## Next Steps

1. Complete 100-epoch training on G5 instance
2. Download and evaluate the trained model
3. Deploy the model to the real-time inference pipeline
4. Document the performance improvements compared to previous models
