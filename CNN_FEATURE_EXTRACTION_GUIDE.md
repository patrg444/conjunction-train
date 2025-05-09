# CNN Feature Extraction Guide

This guide explains how to use the generic extraction and monitoring scripts to process audio emotion datasets through the fixed CNN feature extraction pipeline.

## Overview

The system includes three main components:

1. **Fixed Preprocessing Script**: `fixed_preprocess_cnn_audio_features.py` - The core Python script that corrects shape issues
2. **Generic Extraction Script**: `extract_dataset_cnn_features.sh` - Configurable shell script to run extraction for any dataset
3. **Generic Monitoring Script**: `monitor_dataset_cnn_extraction.sh` - Configurable shell script to track extraction progress

## Using the Generic Extraction Script

The `extract_dataset_cnn_features.sh` script allows you to process any dataset without creating custom scripts for each one.

### Basic Usage

```bash
./extract_dataset_cnn_features.sh --dataset [name] --input [input_dir] --output [output_dir]
```

### Parameters

- `--dataset`: (Optional) Name of the dataset (for display purposes)
- `--input`: (Optional) Path to the input spectrogram directory
- `--output`: (Optional) Path to the output CNN features directory
- `--workers`: (Optional) Number of worker processes (default: 1)
- `--key`: (Optional) Path to SSH key
- `--server`: (Optional) Server address in user@host format

### Examples

1. **Processing CREMA-D dataset:**

```bash
./extract_dataset_cnn_features.sh \
  --dataset "CREMA-D" \
  --input data/crema_d_features_spectrogram \
  --output data/crema_d_features_cnn_fixed
```

2. **Processing RAVDESS dataset:**

```bash
./extract_dataset_cnn_features.sh \
  --dataset "RAVDESS" \
  --input data/ravdess_features_spectrogram \
  --output data/ravdess_features_cnn_fixed
```

3. **Custom configuration with different server:**

```bash
./extract_dataset_cnn_features.sh \
  --dataset "Custom" \
  --input data/custom_features_spectrogram \
  --output data/custom_features_cnn_fixed \
  --workers 2 \
  --key ~/.ssh/custom_key.pem \
  --server user@custom.server.com
```

## Using the Generic Monitoring Script

The `monitor_dataset_cnn_extraction.sh` script allows you to check extraction progress for any dataset.

### Basic Usage

```bash
./monitor_dataset_cnn_extraction.sh --dataset [name] --output [output_dir]
```

### Parameters

- `--dataset`: (Optional) Name of the dataset (for display purposes)
- `--output`: (Optional) Path to the output CNN features directory
- `--key`: (Optional) Path to SSH key
- `--server`: (Optional) Server address in user@host format

### Examples

1. **Monitoring CREMA-D extraction:**

```bash
./monitor_dataset_cnn_extraction.sh \
  --dataset "CREMA-D" \
  --output data/crema_d_features_cnn_fixed
```

2. **Monitoring RAVDESS extraction:**

```bash
./monitor_dataset_cnn_extraction.sh \
  --dataset "RAVDESS" \
  --output data/ravdess_features_cnn_fixed
```

## Working with a New Dataset

To extract CNN features for a new audio emotion dataset:

1. **Prepare spectrograms**: Generate spectrograms for your audio files using your preferred method
2. **Run extraction**: Use the generic extraction script with appropriate parameters
3. **Monitor progress**: Use the generic monitoring script to track extraction progress
4. **Verify results**: Ensure the output files have the correct shape for downstream models

### Example Workflow for a New Dataset

```bash
# Start extraction
./extract_dataset_cnn_features.sh \
  --dataset "NewDataset" \
  --input data/new_dataset_features_spectrogram \
  --output data/new_dataset_features_cnn_fixed \
  --workers 2

# In another terminal, monitor progress
./monitor_dataset_cnn_extraction.sh \
  --dataset "NewDataset" \
  --output data/new_dataset_features_cnn_fixed
```

## Troubleshooting

If you encounter issues during extraction:

1. **Check input shapes**: Ensure your spectrograms have consistent dimensions
2. **Reduce worker count**: Try setting `--workers 1` to debug one file at a time
3. **Check logs**: SSH to the server and check logs for detailed error messages
4. **Disk space**: Ensure sufficient disk space on the server for output files
5. **Memory issues**: For large datasets, consider processing in smaller batches

## Technical Details

The fixed script addresses a shape incompatibility where:
- The CNN model expects 4D input: (None, None, 128, 1)
- Spectrograms are typically 3D: (1, 128, X) where X varies

The fix reshapes the input spectrograms to match the expected shape by adding the necessary channel dimension.
