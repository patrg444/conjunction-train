# Direct AWS Training Management Scripts

These scripts provide direct access to manage your AWS training job without going through the interactive menu.

## Available Scripts

### 1. Upload Fixed Script & Restart Training

```bash
cd aws-setup
./direct_upload_and_restart.sh
```

This uploads the fixed train_branched_6class.py script (with UTF-8 encoding declaration) and restarts the training on the remote instance.

### 2. Check Training Progress

```bash
cd aws-setup
./direct_check_progress.sh
```

This connects to the instance and shows the training log in real-time (press Ctrl+C to exit).

### 3. Monitor System Resources

```bash
cd aws-setup
./direct_monitor_cpu.sh
```

This shows CPU utilization and running processes on the instance.

### 4. Download Results

```bash
cd aws-setup
./direct_download_results.sh
```

This downloads the trained model files to a local 'results' directory.

### 5. Stop or Terminate Instance

```bash
cd aws-setup
./direct_stop_instance.sh stop      # To stop the instance (can restart later)
./direct_stop_instance.sh terminate # To terminate the instance permanently
```

## Common Workflow

1. First, upload the fixed script and restart training:
   ```bash
   ./direct_upload_and_restart.sh
   ```

2. Monitor the training progress:
   ```bash
   ./direct_check_progress.sh
   ```

3. When training is complete, download the results:
   ```bash
   ./direct_download_results.sh
   ```

4. Stop the instance to prevent additional charges:
   ```bash
   ./direct_stop_instance.sh stop
   ```

## Notes

- All scripts use SSH key-based authentication with `emotion-recognition-key-20250322082227.pem`
- The scripts connect as user "ec2-user" to the EC2 instance
- AWS credentials are stored in ~/.aws/credentials for CLI commands
- The instance ID and IP address are hardcoded in the scripts
