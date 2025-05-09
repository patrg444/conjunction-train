# TCN Model Retesting Guide

This guide explains how to retest the TCN models with enhanced logging and validation accuracy tracking using the `retest_tcn_models.sh` script.

## Overview

The `retest_tcn_models.sh` script provides a comprehensive solution for retesting TCN models with improved logging and error handling. It handles the entire process from stopping any existing training, deploying the model with enhanced logging, to setting up monitoring.

## Prerequisites

- SSH access to the AWS instance (IP: 3.235.76.0)
- SSH key file at: `./aws-setup/emotion-recognition-key-fixed-20250323090016.pem`
- The TCN model training script at: `scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py`

## How to Use

Simply run the script from the project directory:

```bash
./retest_tcn_models.sh
```

This will:
1. Stop any existing training processes
2. Deploy the model with enhanced logging
3. Set up a monitoring script
4. Start the training process

## What the Script Does

### 1. Connection Validation
- Validates SSH connection to the AWS instance before proceeding

### 2. Process Cleanup
- Stops any existing training processes
- Removes PID files from previous runs
- Ensures a clean environment for testing

### 3. Model Deployment
- Transfers the TCN model training script to the AWS instance
- Completes the script with enhanced logging configurations:
  - CSV logging of all metrics
  - TensorBoard logging for visualization
  - Separate validation accuracy logging for easy tracking
  - Detailed environment information logging
- Starts the training process with a unique version identifier

### 4. Monitoring Setup
- Creates a custom monitoring script specific to this test run
- Provides real-time tracking of training progress

## Enhanced Logging Features

This script adds several enhanced logging features:

1. **Unique Versioning**: Each run gets a timestamp-based version (e.g., v20250326_153045)
2. **Validation Accuracy Tracking**: Separate logs for validation accuracy
3. **Best Model Tracking**: Records best validation accuracy and corresponding epoch
4. **CSV Metrics**: All training metrics saved in CSV format for easy analysis
5. **TensorBoard Support**: Visualization of training trends
6. **Environment Information**: Python, TensorFlow, and NumPy versions logged

## Monitoring the Training

After running the script, a monitoring script will be created:

```bash
./monitor_tcn_<version>.sh
```

This monitoring script provides:
- Real-time log output
- Process status checks
- Best validation accuracy reporting

## Validation Accuracy Extraction

After training completes, validation accuracy information is available:

1. **Best Validation Accuracy**: Stored in `best_val_accuracy.txt` on the AWS instance
2. **Complete History**: Available in `training_metrics_<version>.csv`
3. **JSON Format**: Complete history in `training_history_<version>.json`

## Troubleshooting

If you encounter issues:

1. **SSH Connection Problems**:
   - Check that the key file exists and has correct permissions (chmod 400)
   - Verify the AWS instance IP address is correct

2. **Script Transfer Issues**:
   - Check that the training script exists locally
   - Ensure SCP can transfer files to the remote instance

3. **Training Crashes**:
   - Check the log file for error messages
   - Verify Python environment on the AWS instance

4. **Monitoring Issues**:
   - Check that the PID file was created correctly
   - Verify the log file is being written to

## Comparing Models

Use the existing comparison tools with the newly generated logs:

```bash
./extract_all_models_val_accuracy.py --plot
./compare_models.sh
```

This will help you compare the retested model with previous versions.
