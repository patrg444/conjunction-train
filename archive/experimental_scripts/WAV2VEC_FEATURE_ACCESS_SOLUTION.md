# WAV2VEC Feature Access Solution

This document explains the solution implemented to fix the issues related to accessing and using WAV2VEC features for emotion recognition model training.

## Problem Summary

The initial attempt to train the ATTN-CRNN model failed due to several issues:

1. The script was attempting to upload features to EC2 even though they were already present.
2. The training script couldn't properly access the WAV2VEC features on the server.
3. The initial implementation encountered a TensorFlow compatibility issue with multiprocessing parameters.
4. Memory consumption was potentially too high when loading all WAV2VEC features at once.

## Solution Components

The solution consists of several key components:

### 1. TensorFlow-Compatible Memory-Efficient Implementation

We created a memory-efficient version of the ATTN-CRNN training script that:

- Uses TensorFlow's Sequence API for efficient batch loading
- Loads WAV2VEC features on-demand rather than all at once
- Removes incompatible parameters from TensorFlow function calls
- Implements proper error handling for feature loading

### 2. Data Access Configuration

We created a symbolic link on the EC2 instance to ensure that:

- The training script has easy access to the WAV2VEC features
- The data directory structure is consistent and manageable
- File permissions are correctly set

### 3. Deployment Scripts

We developed deployment scripts that:

- Upload the fixed training script to the EC2 instance
- Set up the necessary directory structure and symbolic links
- Launch the training process in a tmux session for persistence
- Include proper error handling and status reporting

### 4. Monitoring Tools

We created a comprehensive monitoring script that allows:

- Checking the status of the training process
- Viewing recent training logs
- Monitoring GPU and memory usage
- Verifying model file generation and timestamps
- Attaching to the tmux session for direct interaction

## Usage Guide

### Training Script Deployment

Use one of the following scripts to deploy and start training:

```bash
# Deploy memory-efficient version
./deploy_fixed_attn_crnn_memory.sh

# Deploy TensorFlow-compatible version
./deploy_fixed_attn_crnn_v2.sh
```

### Monitoring Training

Use the monitoring script with various options to track training progress:

```bash
# Show basic usage information
./monitor_attn_crnn_training.sh

# Check training status
./monitor_attn_crnn_training.sh -c

# Show recent training logs
./monitor_attn_crnn_training.sh -l

# Show GPU and memory usage
./monitor_attn_crnn_training.sh -s

# Check if model file has been saved
./monitor_attn_crnn_training.sh -m

# Attach to tmux session to see live output
./monitor_attn_crnn_training.sh -t
```

### Continuous Monitoring

For continuous monitoring, you can use:

```bash
# Stream ATTN-CRNN training output
./stream_attn_crnn_monitor.sh -c
```

## Technical Details

### Memory-Efficient Data Generator

The key to memory efficiency is our custom data generator:

```python
class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator that loads and processes files in batches."""
    
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, max_seq_length=1000):
        # Initialize generator parameters
        
    def __len__(self):
        # Return number of batches
        
    def __getitem__(self, index):
        # Load and preprocess a batch of files
        
    def on_epoch_end(self):
        # Shuffle data between epochs
```

This generator loads files in batches only when needed, rather than keeping all files in memory.

### TensorFlow Compatibility Fix

We removed `workers` and `use_multiprocessing` parameters from the `model.fit()` call, which were causing compatibility issues with the version of TensorFlow running on the EC2 instance:

```python
# Before (causing error):
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=args.epochs, 
    callbacks=callbacks,
    workers=4,
    use_multiprocessing=True
)

# After (fixed):
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=args.epochs,
    callbacks=callbacks
)
```

### Symbolic Link Setup

On the EC2 instance, we ensured the WAV2VEC features directory is properly linked:

```bash
# Create symbolic link if it doesn't exist
if [ ! -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
  ln -sf "/home/ubuntu/audio_emotion/models/wav2vec" /home/ubuntu/emotion_project/wav2vec_features
fi
```

## Next Steps

1. Let the model complete training
2. Evaluate model performance on validation data
3. Generate confusion matrix and classification report
4. Fine-tune hyperparameters if necessary
5. Integrate with other emotion recognition components

The model is now training successfully and should complete the specified number of epochs without memory issues or crashes.
