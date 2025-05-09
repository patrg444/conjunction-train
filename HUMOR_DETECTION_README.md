# Humor Detection Pipeline

This directory contains scripts and tools for downloading, processing, and training text-based humor detection models. The pipeline focuses on using real-world humor data from stand-up comedy and short jokes rather than synthetic examples.

## Overview

The humor detection system is designed to recognize humorous vs. non-humorous text using DistilBERT, a lightweight version of BERT that maintains high accuracy while requiring fewer computational resources. The pipeline supports:

- Downloading and processing the UR-FUNNY dataset (~3800 jokes from stand-up comedy)
- Optional integration with Short-Humor dataset (if available)
- Balancing and merging multiple humor sources
- Training DistilBERT for efficient text-based humor detection

## Datasets

### UR-FUNNY
- **Source**: University of Rochester's dataset of stand-up comedy with manual punch line annotations
- **Size**: ~3800 jokes / 6k clips with binary humor labels
- **Features**: Video + audio + text (we use text only for DistilBERT)
- **Reference**: Hasan et al. (2019) - "UR-FUNNY: A Multimodal Language Dataset for Understanding Humor"

### Short-Humor (Optional)
- Text-based dataset of humorous/non-humorous content
- Useful for supplementing the stand-up comedy data

## Scripts

### Main Scripts

1. **download_ur_funny_v2.py**
   - Downloads UR-FUNNY V2 dataset from official Dropbox link 
   - Extracts pickle files containing transcripts and humor annotations
   - Processes data into standardized manifest format
   - Creates train/val splits using the dataset's predefined folds

2. **download_short_humor.py**
   - Downloads and processes Short-Humor dataset (optional)

3. **scripts/merge_humor_manifests.py**
   - Combines multiple humor datasets into a single manifest
   - Balances positive/negative examples across sources
   - Ensures proper representation from each dataset

4. **train_humor_workflow.sh**
   - One-stop script that orchestrates the entire pipeline
   - Handles dependencies, downloads, processing, and training

### Support Scripts

- **enhanced_train_distil_humor.py**: Main training script for DistilBERT
- **train_distilbert_ec2.sh**: Helper script for training on EC2 instances

## Getting Started

The simplest way to get started is to run the workflow script:

```bash
# Make the script executable
chmod +x train_humor_workflow.sh

# Run the workflow
./train_humor_workflow.sh
```

This will:
1. Install necessary dependencies
2. Download the UR-FUNNY dataset
3. (Optionally) download Short-Humor dataset
4. Process and merge manifests
5. Offer to start training

## Manual Steps

If you prefer to run each step individually:

```bash
# 1. Install dependencies
pip install pandas numpy tqdm requests gitpython

# 2. Download and process UR-FUNNY
python download_ur_funny_v2.py

# 3. (Optional) Download and process Short-Humor
pip install nltk
python download_short_humor.py

# 4. Merge manifests
python scripts/merge_humor_manifests.py

# 5. Train the model
python enhanced_train_distil_humor.py \
    --train_manifest datasets/manifests/humor/combined_train_humor.csv \
    --val_manifest datasets/manifests/humor/combined_val_humor.csv
```

## Directory Structure

After running the scripts, you'll have:

```
/datasets
  /humor_datasets
    /ur_funny            # Raw UR-FUNNY data
  /manifests
    /humor
      ur_funny_train_humor.csv      # Train manifest for UR-FUNNY
      ur_funny_val_humor.csv        # Validation manifest
      short_humor_train_humor.csv   # (If you downloaded Short-Humor)
      short_humor_val_humor.csv     # (If you downloaded Short-Humor)
      combined_train_humor.csv      # Merged training manifest
      combined_val_humor.csv        # Merged validation manifest
```

## Training Parameters

The default training uses DistilBERT with the following parameters:
- Learning rate: 5e-5
- Batch size: 16
- Max sequence length: 128
- Epochs: 3

You can customize these parameters by editing `enhanced_train_distil_humor.py` or passing command-line arguments.

## References

- Hasan, M. K., Rahman, W., Bagher Zadeh, A., Zhong, J., Tanveer, M. I., Morency, L. P., & Hoque, M. E. (2019). UR-Funny: A multimodal language dataset for understanding humor. In Proceedings of EMNLP-IJCNLP.
