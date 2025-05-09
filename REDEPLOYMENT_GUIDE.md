# TCN Model Redeployment Guide

This guide explains how to use the `redeploy_all_tcn_models.sh` script to sequentially redeploy all TCN models on AWS.

## Overview

The redeployment script automates the process of redeploying multiple TCN models one after another on your AWS instance. It handles:

- Stopping any existing training processes
- Deploying models in sequence with proper waiting periods
- Monitoring initial training to ensure proper startup
- Checking validation accuracy
- Logging all activities for later review

## Models Included

The script redeploys the following models:

1. **Fixed TCN Large Model** (`deploy_fixed_tcn_large_model.sh`)
2. **Fixed TCN Large Model Simple** (`deploy_fixed_tcn_large_model_simple.sh`) 
3. **Fixed TCN Model v2** (`deploy_fixed_tcn_model_v2.sh`)

## Prerequisites

- SSH access to the AWS instance (IP: 3.235.76.0)
- SSH key file at: `./aws-setup/emotion-recognition-key-fixed-20250323090016.pem`
- All deployment scripts must be present in the current directory

## How to Use

Simply run the script from the project directory:

```bash
./redeploy_all_tcn_models.sh
```

This will:
1. Validate SSH connection to the AWS instance
2. Stop any existing training processes
3. Deploy each model in sequence
4. Wait 30 minutes between deployments
5. Monitor each model briefly to ensure proper startup
6. Check for initial validation accuracy
7. Create detailed logs in a timestamped directory

## How It Works

### 1. Initialization
The script creates a timestamped log directory for all deployment logs, such as `redeployment_logs_20250326_154500/`.

### 2. Process Cleanup
It first ensures all existing training processes are stopped by:
- Running the existing `stop_tcn_large_training.sh` script
- Using additional cleanup commands to kill any remaining Python processes
- Removing any PID files that might be lingering

### 3. Sequential Deployment
For each model:
- The appropriate deployment script is executed
- Outputs are logged to the log directory
- Associated monitoring scripts are identified and copied to the log directory
- There's a 60-second waiting period for initialization
- Brief monitoring is performed to ensure the model is running
- Initial validation accuracy is checked
- A 30-minute waiting period is observed before deploying the next model (except after the last model)

### 4. Final Summary
The script provides a summary of the deployment operations and guidance on how to monitor the training progress.

## Log Directory Structure

The redeployment process creates a log directory with the following files:

- `redeployment_summary.log` - Overall timing information
- `<model_name>_deployment.log` - Deployment script output for each model
- `<model_name>_monitoring.log` - Initial monitoring information
- `<model_name>_accuracy.log` - Initial validation accuracy data
- `<model_name>_monitor.sh` - Monitoring scripts for each deployed model

## Monitoring Deployed Models

There are three ways to monitor the deployed models:

1. **Using model-specific monitoring scripts**: 
   ```bash
   ./<log_directory>/<model_name>_monitor.sh
   ```

2. **Using the general cross-platform monitoring script**:
   ```bash
   ./continuous_tcn_monitoring_crossplatform.sh
   ```

3. **Using the validation accuracy extraction tools**:
   ```bash
   ./get_all_model_accuracies.sh
   ./compare_models.sh
   ```

## Troubleshooting

If you encounter issues:

1. **SSH Connection Problems**:
   - Check that the key file exists and has correct permissions (`chmod 400`)
   - Verify the AWS instance IP address is correct

2. **Script Execution Failures**:
   - Check the deployment logs for specific error messages
   - Verify that the AWS instance has sufficient resources

3. **Training Not Starting**:
   - Check if Python is installed and functioning on the AWS instance
   - Verify that dependencies are properly installed

4. **Validation Accuracy Not Appearing**:
   - It may take several epochs before validation accuracy is computed
   - Check if the training data is being properly loaded

## Related Tools

- `retest_tcn_models.sh` - For individual model retesting with enhanced logging
- `extract_all_models_val_accuracy.py` - For extracting validation accuracy from all models
- `compare_model_accuracies.py` - For creating comparison visualizations

## Important Notes

- Models are trained sequentially, meaning each model will train to completion before the next one starts
- The entire redeployment process may take several days to complete
- You can monitor the progress at any time using the monitoring scripts
- If you need to stop a training, use the `stop_tcn_large_training.sh` script
