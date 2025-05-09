# WAV2VEC Feature Detection Fix

## Problem Overview

The WAV2VEC feature files were being repeatedly uploaded to the EC2 instance even though they already existed on the server. This was occurring because the detection scripts (`wav2vec_data_setup.sh` and `stream_attn_crnn_monitor.sh`) were only looking for feature files in a single directory path: `/home/ubuntu/emotion_project`.

The directory scan revealed that WAV2VEC feature files actually exist in multiple locations on the EC2 instance:

| Location | Feature Count |
|----------|--------------|
| /home/ubuntu/audio_emotion/models/wav2vec | 8,690 files |
| /home/ubuntu/emotion-recognition/crema_d_features_facenet | 7,441 files |
| /home/ubuntu/emotion-recognition/npz_files/CREMA-D | 7,441 files |
| /home/ubuntu/emotion-recognition/crema_d_features_audio | 530 files |

Since the original detection script only checked `/home/ubuntu/emotion_project`, it would report "0 WAV2VEC feature files found" even though there were tens of thousands of feature files in other directories.

## Solution Implemented

1. Updated `wav2vec_data_setup.sh` to check multiple directories for WAV2VEC features:
   - The original emotion_project directory
   - The wav2vec_features subdirectory
   - The dataset-specific features directory
   - The audio_emotion/models/wav2vec directory (8,690 files)
   - The emotion-recognition directories with various feature files

2. Updated `stream_attn_crnn_monitor.sh` with the same expanded directory checks
   - This ensures consistent reporting between the setup and monitoring scripts
   - The health check will now correctly report all WAV2VEC feature files

3. Created a verification script `verify_wav2vec_feature_check.sh` that:
   - Scans all potential locations for WAV2VEC features
   - Reports the file count in each location
   - Displays the total feature count across all directories
   - Indicates when directories exist but have no .npz files

## Benefits

- **No More Redundant Uploads**: The system will now correctly detect existing feature files and avoid uploading them again, saving bandwidth and time.
  
- **Accurate Health Monitoring**: The monitoring script will show the true count of feature files, providing better visibility into the system state.

- **Improved Directory Support**: The updated scripts now handle feature files stored in multiple directories and subdirectories.

## How to Verify

Run the verification script to confirm feature detection without making any changes:

```bash
./verify_wav2vec_feature_check.sh
```

This will display a table showing all directories that contain WAV2VEC features and their respective file counts.

## Technical Details

The core issue was in the feature detection function, which has been expanded to scan multiple directories:

```bash
# Define possible locations where feature files might exist
POSSIBLE_DIRS=(
  "$REMOTE_DIR"                                     # Main project directory
  "$REMOTE_DIR/wav2vec_features"                    # Subdirectory for features
  "$REMOTE_DIR/${LOCAL_DATASET}_features"           # Dataset-specific directory
  "/home/ubuntu/${LOCAL_DATASET}_features"          # Alternate location
  "/home/ubuntu/audio_emotion/models/wav2vec"       # Primary wav2vec location (8690 files)
  "/home/ubuntu/emotion-recognition/crema_d_features_facenet" # FaceNet features
  "/home/ubuntu/emotion-recognition/npz_files/CREMA-D"        # Another CREMA-D location
  "/home/ubuntu/emotion-recognition/crema_d_features_audio"   # Audio features
)
```

The script now checks each of these locations and accumulates the total count of feature files.
