# TCN Large Model Training Monitoring Guide

This guide explains how to use the advanced training monitoring tools created for the TCN Large Model training process.

## Monitoring Scripts

Two monitoring scripts are provided to track the TCN large model training progress:

1. **`continuous_tcn_monitoring.sh`** - Standard monitoring script (macOS optimized)
2. **`continuous_tcn_monitoring_crossplatform.sh`** - Cross-platform version (works on both macOS and Linux)

## Features

These monitoring scripts provide:

- **Real-time metrics** - Validation accuracy, learning rate, and training progress
- **System resource tracking** - CPU, memory, and disk usage
- **Best performance tracking** - Records and displays the best validation accuracy
- **Training stall detection** - Alerts when training has not progressed for over 90 minutes
- **Error monitoring** - Detects and reports errors in the training log
- **Process monitoring** - Verifies that the training process is still active
- **Auto-refresh** - Updates metrics every 30 seconds (configurable)

## Prerequisites

- SSH access to the AWS EC2 instance running the training job
- SSH key file: `./aws-setup/emotion-recognition-key-fixed-20250323090016.pem`

## Usage

### 1. Deploy the Fixed TCN Model (if not done already)

```bash
# Make sure deployment script is executable
chmod +x deploy_fixed_tcn_large_model_simple.sh

# Deploy the model to AWS
./deploy_fixed_tcn_large_model_simple.sh
```

### 2. Start Continuous Monitoring

```bash
# Run the monitoring script (choose one)

# For macOS
./continuous_tcn_monitoring.sh

# For any platform (macOS or Linux)
./continuous_tcn_monitoring_crossplatform.sh
```

### 3. Understanding the Output

The monitoring script will display:

- **Training Summary**
  - Epochs completed
  - Current validation accuracy
  - Best validation accuracy achieved
  - Current learning rate
  - Training duration
  - Error detection

- **Recent Training Progress**
  - Latest epoch metrics
  - Latest validation accuracy

- **System Status**
  - CPU, memory, and disk usage on the EC2 instance

- **Log Tail**
  - Last 10 lines of the training log

## Customization

Both scripts have configurable parameters at the top:

- `MONITORING_INTERVAL` - Seconds between update cycles (default: 30)
- `ALERT_THRESHOLD` - Minutes without progress before alerting (default: 90)

You can adjust these parameters based on your monitoring needs.

## Manual Monitoring

If you prefer a simpler approach, you can also monitor the training log directly:

```bash
# SSH into the EC2 instance
ssh -i ./aws-setup/emotion-recognition-key-fixed-20250323090016.pem ec2-user@3.235.76.0

# Monitor the log file
tail -f ~/emotion_training/training_branched_regularization_sync_aug_tcn_large_fixed.log
```

## Troubleshooting

If the monitoring script reports that the training has stopped unexpectedly, check:

1. EC2 instance status - Ensure the instance is still running
2. CPU/Memory utilization - Very high usage might indicate resource exhaustion
3. Check the full log for errors:
   ```bash
   ssh -i ./aws-setup/emotion-recognition-key-fixed-20250323090016.pem ec2-user@3.235.76.0 "grep -E 'ERROR|Exception|Traceback' ~/emotion_training/training_branched_regularization_sync_aug_tcn_large_fixed.log"
   ```

## Expected Results

With the improved optimization parameters in the fixed TCN model, we expect:

1. Stable training progression
2. Improved validation accuracy (targeting >85%)
3. Better generalization compared to the original model that was stuck at 83.8%
