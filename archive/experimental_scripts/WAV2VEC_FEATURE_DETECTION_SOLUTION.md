# WAV2VEC Feature Detection Issue & Solution

## Problem Overview

When running the ATTN-CRNN training pipeline, despite WAV2VEC features already existing on the EC2 instance, they weren't being detected properly, resulting in unnecessary uploads. This document explains why this happens and how our solution addresses it.

## Root Causes

1. **Path Mismatch**: The ATTN-CRNN v2 model is looking for features in specific paths (`/data/wav2vec_features` and `/data/wav2vec_crema_d`), but the features exist in different legacy locations on the EC2 instance.

2. **Inadequate Feature Detection**: The original script (`wav2vec_data_setup.sh`) only checks a limited set of paths and doesn't properly handle the case where features exist but in different locations.

3. **No Symlinking**: There was no mechanism to create symbolic links from the existing feature locations to the directories expected by the training scripts.

4. **Weak Access Verification**: The script didn't verify if the features could actually be accessed by the current user with appropriate permissions.

## Evidence

From the monitoring logs, we can see:

```
=== ATTN-CRNN TRAINING HEALTH CHECK ===
Checking training status...
✓ tmux session 'audio_train' is active
✗ WARNING: training process not running
WAV2VEC feature files found: 0
```

Despite the features existing on the EC2 instance, the monitor reports "WAV2VEC feature files found: 0" because it's looking in specific directories based on the ATTN-CRNN v2 model's expectations.

## Solution: Improved Detection and Symlinking

The `improved_wav2vec_data_setup.sh` script addresses these issues by:

1. **Comprehensive Path Checking**: It checks multiple locations where WAV2VEC features might exist, including both the target paths expected by the ATTN-CRNN v2 model and legacy paths used by previous models.

2. **Smart Decision Making**: Based on where features are found, it determines the appropriate action:
   - If features exist in target directories: No action needed
   - If features exist only in legacy directories: Create symlinks
   - If features exist in both: Create additional symlinks as needed
   - If no features exist: Upload from local directory

3. **Permission Verification**: The script verifies and adjusts permissions as needed to ensure features are accessible.

4. **Automatic Symlinking**: Instead of moving or copying large feature files (which is slow and wastes disk space), it creates symbolic links from existing files to the expected locations.

## Implementation Details

The script handles four distinct cases, returning different status codes:

- **Status 0**: No features found anywhere → Upload features from local machine
- **Status 1**: Features already exist in target directories → No action needed
- **Status 2**: Features exist only in legacy directories → Create symlinks to target directories
- **Status 3**: Features exist in both target and legacy directories → Create additional symlinks

For each legacy directory containing features, the script creates symbolic links to the appropriate target directory, maintaining the file organization expected by the ATTN-CRNN v2 model.

## How to Use

Run the improved script before launching the training:

```bash
./improved_wav2vec_data_setup.sh
```

The script will:
1. Check for existing features on the EC2 instance
2. Create symlinks if needed
3. Only upload features if absolutely necessary
4. Report detailed findings to help diagnose any issues

After running this script, you can proceed with the ATTN-CRNN v2 training using:

```bash
./deploy_attn_crnn_v2.sh
```

And monitor progress with:

```bash
./monitor_attn_crnn_v2.sh -c
```

## Advantages Over Previous Approach

- **Avoids Redundant Uploads**: Prevents uploading features that already exist
- **Saves Bandwidth and Time**: No need to wait for large uploads when unnecessary
- **Conserves Disk Space**: Uses symlinks instead of creating duplicate copies
- **More Robust**: Better handles different directory structures and edge cases
- **Better Feedback**: Provides clear information about what was found and what actions were taken

By properly detecting existing features and creating symbolic links where needed, this solution eliminates the redundant uploading of WAV2VEC features while ensuring the training scripts can find them in the expected locations.
