# Facenet Training Fix: Missing Feature Files on Server

## Problem Description

The Facenet video-only emotion recognition model deployment failed to train properly because the feature files, while available locally, were not uploaded to the AWS server. This resulted in the error:

```
ValueError: The PyDataset has length 0
```

The monitoring script showed no running training process and no tmux sessions, indicating the training had failed to start or terminated almost immediately.

## Root Cause Analysis

Upon investigation, we found:

1. The deployment script (`deploy_full_facenet_training.sh`) successfully uploaded the training code and scripts but did not include the feature files.
2. The feature files exist locally in the `crema_d_features_facenet/` directory (containing files like `1078_IWL_ANG_XX.npz` with emotion labels).
3. When the training script ran on the server, it couldn't find any valid feature files in the expected directory, resulting in a dataset with length 0.

## Solution

We created a dedicated script (`upload_facenet_features_and_restart.sh`) that:

1. Uploads the feature files tarball (558MB) to the server
2. Extracts the tarball into the correct directory
3. Stops any existing failed training processes
4. Restarts the training in a new tmux session

This approach ensures all necessary files are in place before training begins.

## Deployment Instructions

To fix the training deployment:

1. Run the script:
   ```bash
   ./upload_facenet_features_and_restart.sh
   ```

2. Monitor the training progress:
   ```bash
   ./facenet_monitor_helper.sh
   ```

3. To view live training output:
   ```bash
   ssh -i /Users/patrickgloria/Downloads/gpu-key.pem ubuntu@18.208.166.91
   tmux attach -t facenet_training
   ```
   (Use Ctrl+B, D to detach from the tmux session without stopping it)

## Verification

After running the fix script, verify the training is working by checking:

1. The feature files are properly extracted:
   ```bash
   ssh -i /Users/patrickgloria/Downloads/gpu-key.pem ubuntu@18.208.166.91 "ls -la /home/ubuntu/emotion-recognition/crema_d_features_facenet"
   ```

2. The training process is running:
   ```bash
   ./facenet_monitor_helper.sh
   ```

3. The training log shows successful batch processing:
   ```bash
   ssh -i /Users/patrickgloria/Downloads/gpu-key.pem ubuntu@18.208.166.91 "tail -n 20 /home/ubuntu/emotion-recognition/facenet_full_training/training_output.log"
   ```

## Preventive Measures for Future Deployments

To prevent similar issues in the future:

1. **Include Data in Deployment**: Modify deployment scripts to include required data files, or add explicit checks to ensure needed data is available.

2. **Add Pre-training Validation**: Implement validation steps at the start of training scripts to check for required directories and minimum numbers of data files.

3. **Enhance Monitoring**: Add more robust monitoring to detect training failures, including data availability checks.

4. **Separation of Concerns**: Create separate scripts for code deployment and data deployment to manage large file transfers more effectively.

5. **Improved Error Handling**: Enhance error messages to clearly indicate when training fails due to missing data.

## References

- `upload_facenet_features_and_restart.sh` - Script to fix the current deployment
- `facenet_monitor_helper.sh` - Script to monitor training status
- `deploy_full_facenet_training.sh` - Original deployment script (missing data upload)
- `FACENET_VIDEO_ONLY_FIX_COMPLETE.md` - Documentation of the Facenet generator fixes
