# EC2 Instance Dataset Input File Locations

This document outlines the identified locations for input data files and directories specifically on the EC2 instance `ubuntu@52.90.218.245`, based on SSH exploration. Note that this structure differs significantly from the local project layout documented in `DATASET_LOCATIONS.md`.

## Raw Data Directories

*   `/home/ubuntu/datasets/`: This directory contains the primary raw datasets.
    *   `crema_d_videos/` (Directory containing CREMA-D videos)
    *   `ravdess_videos/` (Directory containing RAVDESS videos)
    *   `video_manifest.csv` (Manifest file, likely generated from the above directories)
    *   `video_manifest/` (Directory, contents unknown)

## Manifest Files

*   `/home/ubuntu/conjunction-train/splits/`: Contains manifests used by Hubert SER and Fusion models.
    *   `train.csv`, `val.csv`, `test.csv`
    *   `crema_d_train.csv`, `crema_d_val.csv`
*   `/home/ubuntu/datasets/video_manifest.csv`: Manifest for video datasets (SlowFast, R3D).
*   *Laughter Manifest (`laughter_v1.csv`):* Location unclear on EC2. Expected in `.../datasets_raw/manifests/` based on local structure, but that path wasn't found within `/home/ubuntu/conjunction-train/`. May exist elsewhere under `/home/ubuntu/` or need creation/upload.

## Feature Directories & Files

Feature locations on EC2 are distributed and differ from the local setup:

*   **Wav2Vec Features:**
    *   `/data/wav2vec_features/`
    *   `/data/wav2vec_crema_d/`
    *   *(Note: These directories likely contain the actual features, potentially populated via symlinks from other locations like `/home/ubuntu/audio_emotion/models/wav2vec/` or `/home/ubuntu/emotion_project/wav2vec_features/`, although `/home/ubuntu/audio_emotion/` was not found).*
*   **Hubert Embeddings:**
    *   `/home/ubuntu/conjunction-train/splits/`: Contains the precomputed Hubert embeddings (`.npz` files) corresponding to the manifests in the same directory.
        *   `train_embeddings.npz`, `val_embeddings.npz`, `test_embeddings.npz`
        *   `crema_d_train_embeddings.npz`, `crema_d_val_embeddings.npz`
*   **FaceNet Features:**
    *   `/home/ubuntu/conjunction-train/`: Contains CREMA-D FaceNet features (`.npz` files like `1001_DFA_ANG_XX.npz`).
    *   `/home/ubuntu/emotion-recognition/`: Contains RAVDESS FaceNet features within `Actor_*` subdirectories.
*   **CNN Audio & Spectrogram Features:**
    *   `/home/ubuntu/emotion-recognition/data/`: Contains the actual feature directories.
        *   `crema_d_features_cnn_audio/`, `crema_d_features_cnn_fixed/`
        *   `ravdess_features_cnn_audio/`, `ravdess_features_cnn_fixed/`
        *   `crema_d_features_spectrogram/`, `ravdess_features_spectrogram/`
        *   *(Note: Symlinks to these directories exist in `/home/ubuntu/emotion_project/data/`).*

## Normalization Statistics Files

*   Location not explicitly confirmed via `ls`, but likely expected in `/home/ubuntu/conjunction-train/` (like local `audio_mean.npy`, `audio_std.npy`) or within specific model output directories under `/home/ubuntu/` (e.g., `/home/ubuntu/models/`, `/home/ubuntu/emotion_project/models/`). Pattern `models/*normalization_stats.pkl` seen in scripts.

## Configuration Files

*   `/home/ubuntu/conjunction-train/config/`: Contains configuration files like `eGeMAPSv02.conf` and `slowfast_face.yaml`.

## Other Potentially Relevant EC2 Directories

These directories exist under `/home/ubuntu/` and might contain relevant code, scripts, logs, or outputs.

*   `/home/ubuntu/conjunction-train/` (Main project directory)
*   `/home/ubuntu/emotion_project/` (Contains scripts, logs, models, checkpoints, and symlinks to features)
*   `/home/ubuntu/emotion-recognition/` (Contains scripts, logs, models, and actual CNN/Spectrogram/FaceNet features)
*   `/home/ubuntu/emotion_cmp/`
*   `/home/ubuntu/emotion_full_video/`
*   `/home/ubuntu/emotion_fusion/`
*   `/home/ubuntu/emotion_fusion_output/`
*   `/home/ubuntu/emotion_slowfast/`
*   `/home/ubuntu/hubert_large_features/`
*   `/home/ubuntu/models/`
*   `/home/ubuntu/checkpoints/` (Though not listed directly in `/home/ubuntu/`, often created by training scripts within project subdirs)
*   `/home/ubuntu/logs/`
*   `/home/ubuntu/wav2vec_sample/`

**Note:** This documentation reflects the state of the EC2 instance `52.90.218.245` as explored via SSH commands. File locations, especially for features, can vary depending on script execution, symlinks, and specific deployment steps. Always refer to the specific training/processing scripts and their execution context for definitive input paths.
