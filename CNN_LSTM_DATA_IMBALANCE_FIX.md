# CNN-LSTM Data Imbalance Issue

## Problem Identified

After fixing the syntax issues in `train_spectrogram_cnn_pooling_lstm.py`, we've discovered a critical data imbalance issue that would prevent the model from training properly:

### Data Imbalance Analysis

1. **Raw Dataset Sizes:**
   - RAVDESS: 1,472 files
   - CREMA-D: 7,442 files

2. **Processed CNN Features:**
   - `data/ravdess_features_cnn_fixed`: Only **27 files** (1.8% of RAVDESS data)
   - `data/crema_d_features_cnn_fixed`: **7,445 files** (100% of CREMA-D data)

This extreme imbalance (27 vs 7,445) means the model would essentially be trained almost exclusively on CREMA-D data, with minimal representation from RAVDESS. This would create a biased model that likely performs poorly on RAVDESS test data.

## Root Cause

The preprocessing script `preprocess_cnn_audio_features.py` uses output directories named with `*_cnn_audio` suffix, but the model is looking for directories with `*_cnn_fixed` suffix. This suggests there was an interrupted or failed feature extraction run specifically for RAVDESS when creating the "fixed" features.

## Solution

I've created `extract_ravdess_cnn_features.sh` to specifically regenerate the RAVDESS CNN features into the correct directory. This script:

1. Creates a temporary modified version of the preprocessing script
2. Configures it to only process RAVDESS data
3. Directs output to the correct `ravdess_features_cnn_fixed` directory 
4. Runs the processing and verifies the results

## Execution Instructions

1. Make the script executable: `chmod +x extract_ravdess_cnn_features.sh`
2. Run the script: `./extract_ravdess_cnn_features.sh`
3. Once complete, verify that all RAVDESS files have been processed (should be around 1,472 files)
4. Run the CNN-LSTM model training with the balanced datasets

## Expected Improvement

With properly balanced datasets (approx. 1,472 RAVDESS files and 7,445 CREMA-D files), the model should:
- Train on a more diverse set of samples
- Learn features relevant to both datasets
- Achieve better generalization across different emotional speech patterns
- Show improved performance on validation and test sets

The syntax fixes in `train_spectrogram_cnn_pooling_lstm.py` were necessary but insufficient to make the model work correctly. This data balancing fix addresses the underlying issue that was preventing successful training.
