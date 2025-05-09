# Enhanced Wav2Vec Feature Extraction

This document describes the improvements made to the wav2vec feature extraction process for the RAVDESS and CREMA-D datasets.

## Problem Identified

The original wav2vec extraction script (`extract_audio_and_wav2vec_fusion.py`) had two key issues:

1. It relied on the `moviepy.editor` module, which caused import errors on the EC2 instance
2. It only looked for `.mp4` files, but the CREMA-D dataset might contain files with other extensions (like `.flv`)

## Solution Implemented

We developed an enhanced extraction pipeline that addresses these issues:

1. **Direct FFMPEG Usage**: Replaced `moviepy.editor` with direct `ffmpeg` calls via `subprocess` for better reliability
2. **Multi-Format Support**: Added support for multiple video file formats (MP4, FLV, AVI, MOV, etc.)
3. **Self-Diagnostics**: Added diagnostic capabilities to identify and report file types in the dataset directories
4. **Graceful Fallbacks**: If specific file formats aren't found, the script attempts to detect and process any available video formats

## Enhanced Scripts

Two key files have been updated/created:

1. **`fixed_extract_wav2vec.py`**
   - Enhanced to work with multiple video file formats
   - Better error handling and reporting
   - Includes diagnostic capabilities to identify available file types
   - Uses FFMPEG directly instead of moviepy for more reliable audio extraction

2. **`deploy_enhanced_wav2vec_extraction_to_ec2.sh`**
   - Deploys the enhanced extraction script to EC2
   - Includes expanded diagnostics for troubleshooting
   - Provides clearer progress information and results

## Usage

To use the enhanced extraction pipeline:

```bash
# Make the deployment script executable (if not already)
chmod +x deploy_enhanced_wav2vec_extraction_to_ec2.sh

# Run the deployment script
./deploy_enhanced_wav2vec_extraction_to_ec2.sh
```

This will:
1. Upload the enhanced extraction script to the EC2 instance
2. Create and upload a runner script that includes diagnostic capabilities
3. Run the extraction process on a sample of files (default: 10 samples)
4. Report the file types found in each dataset directory
5. Provide instructions for accessing the extracted features

## Expected Outcome

- Successful processing of both RAVDESS and CREMA-D datasets regardless of file extensions
- Wav2vec features extracted and saved as NPZ files
- Fusion model configuration created for later use with SlowFast video model
- Comprehensive diagnostics showing the file types found in each dataset

## After Extraction

Once features are extracted, they can be:
1. Downloaded to your local machine for analysis
2. Used to train a wav2vec-based audio emotion recognition model
3. Combined with the SlowFast video model using the generated fusion configuration
