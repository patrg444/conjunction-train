# LSTM Attention Model Deployment Guide

This guide explains how to deploy, monitor, and manage the LSTM attention model training on AWS EC2 using our optimized approach.

## Overview

The LSTM attention model combines:
- Bi-directional LSTM layers for temporal modeling of both audio and video data
- Attention mechanisms to focus on the most relevant parts of each modality
- Focal loss with class weighting to handle class imbalance
- Multimodal fusion of audio and video features

## Deployment

The model training is deployed on AWS EC2 using the following steps:

1. **Optimized Deployment**: Use the `deploy_lstm_attention_c5_optimized.sh` script to:
   - Create a new c5.24xlarge EC2 instance (96 vCPUs)
   - Set up TensorFlow and dependencies with proper versions
   - Upload only essential files to reduce deployment size and time
   - Configure environment for optimal CPU utilization

```bash
# Deploy with optimized settings
./aws-setup/deploy_lstm_attention_c5_optimized.sh
```

## Monitoring

We provide multiple monitoring options for keeping track of the training process:

### Basic Monitoring

```bash
# Monitor training logs in real-time
./aws-setup/monitor_lstm_attention_model.sh

# Monitor CPU usage
./aws-setup/monitor_lstm_attention_model.sh --cpu
```

### Continuous Monitoring with Auto-Reconnection

For more robust monitoring that can handle temporary connection issues, we created a continuous monitoring script that:
- Automatically reconnects if the connection drops
- Checks both instance and training process status
- Reports CPU, memory, and disk usage
- Shows model checkpoint generation
- Provides training log updates

```bash
# Continuous monitoring with periodic status updates
./aws-setup/continuous_training_monitor.sh

# Continuous monitoring with frequent log updates
./aws-setup/continuous_training_monitor.sh --log

# Quick one-time status report
./aws-setup/continuous_training_monitor.sh --quick
```

## Common Issues and Solutions

1. **OpenSSL/urllib3 Compatibility Issue**:
   - Symptom: `ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips'`
   - Solution: We've fixed this by pinning urllib3 to version 1.26.6 in the deployment script

2. **GLIBCXX Version Issues**:
   - Symptom: `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`
   - Solution: Our deployment uses compatible library versions to avoid this error

3. **Connection Drops**:
   - Problem: SSH connections to EC2 may drop during long monitoring sessions
   - Solution: Use the `continuous_training_monitor.sh` script which automatically reconnects

## Downloading Results

When training is complete, download the model results:

```bash
./aws-setup/download_lstm_attention_results.sh
```

This will download both the trained model and training logs to a local directory.

## Instance Management

- **Status Check**: `aws ec2 describe-instances --instance-ids <instance-id>`
- **Stop Instance**: `aws ec2 stop-instances --instance-ids <instance-id>`
- **Terminate Instance**: `aws ec2 terminate-instances --instance-ids <instance-id>`

## Connection Details

Connection details are stored in `aws-setup/lstm_attention_model_connection.txt` and include:
- Instance ID
- Instance IP
- SSH key location
- Log file path

This file is automatically sourced by the monitoring scripts.

## Troubleshooting

1. **SSH Connection Fails**:
   - Check EC2 security groups allow SSH access
   - Verify the instance is running
   - Ensure the key has the right permissions (`chmod 400`)

2. **Training Process Not Running**:
   - Check training logs for errors
   - Use `continuous_training_monitor.sh --quick` to see if the process is still active

3. **Disk Space Issues**:
   - Use `continuous_training_monitor.sh` to monitor disk usage
   - Consider increasing the EBS volume size if needed
