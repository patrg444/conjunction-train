# Dataset Upload Instructions

This guide explains how to upload the RAVDESS and CREMA-D datasets from your local machine to your EC2 instance using rsync.

## Overview

The `upload_datasets_rsync.sh` script uploads the video files directly from your local machine to the EC2 instance using rsync. This method:

- Provides resumable uploads (if connection drops, it will continue where it left off)
- Verifies file integrity during transfer
- Shows progress during the upload
- Works with the existing dataset files in your local directories

## Prerequisites

1. SSH key file for your EC2 instance (e.g., `gpu-key.pem`)
2. Your EC2 instance address
3. The datasets already exist in your local directories:
   - RAVDESS: `./downsampled_videos/RAVDESS/`
   - CREMA-D: `./downsampled_videos/CREMA-D-audio-complete/`

## Step 1: Configure the upload script

Edit the `upload_datasets_rsync.sh` script and update these variables at the top:

```bash
export KEY=~/Downloads/gpu-key.pem  # Path to your EC2 SSH key file
export EC2=ubuntu@54.162.134.77     # Your EC2 instance address
```

## Step 2: Make the script executable

```bash
chmod +x upload_datasets_rsync.sh
```

## Step 3: Run the script

```bash
./upload_datasets_rsync.sh
```

The script will:
1. Create the necessary directories on the EC2 instance
2. Upload the RAVDESS dataset with resume support
3. Upload the CREMA-D dataset with resume support
4. Verify the upload by counting files on the EC2 instance

## What to expect

- The uploads may take a while depending on your internet connection speed (RAVDESS ≈ 2.8 GB, CREMA-D ≈ 15 GB)
- If your connection drops, simply run the script again - it will resume from where it left off
- Upon completion, you'll see a count of files uploaded to verify everything transferred correctly

## Troubleshooting

- **SSH Permission Denied**: Make sure your key file has the correct permissions: `chmod 400 ~/Downloads/gpu-key.pem`
- **Connection Issues**: If you face connection issues, try running the script with a more stable internet connection
- **Verification Errors**: If the file count doesn't match expectations, run the script again to ensure all files are transferred

## After Upload

Once the datasets are uploaded, they will be available on your EC2 instance at:
- RAVDESS: `/home/ubuntu/datasets/ravdess_videos/`
- CREMA-D: `/home/ubuntu/datasets/crema_d_videos/`

You can now proceed with your emotion recognition model training or feature extraction.
