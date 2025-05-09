# Talk-Level Humor Dataset

## Problem Background

Our initial humor manifests had two main issues:
1. They were referencing the CelebA facial dataset as 'smile', causing confusion with the SMILE humor dataset.
2. The data was segmented at utterance level, which made it difficult for models to capture the full context of humor.

## Solution

We've created a pipeline to build proper talk-level humor manifests from the SMILE dataset:

1. **Data Integration**: We extract text transcripts from the SMILE dataset's multimodal JSON files.
2. **Talk-Level Consolidation**: Rather than using individual utterances, we merge all utterances from a talk into a single transcript to provide full context.
3. **Clear Labeling**: Each talk is assigned a single humor label (1 if humorous, 0 if not).
4. **Dataset Integrity**: We ensure no overlap between training and validation sets to prevent data leakage.

## Dataset Structure

The talk-level manifests are stored in:
- `/datasets/manifests/humor/talk_level_train_humor.csv`
- `/datasets/manifests/humor/talk_level_val_humor.csv`

Each manifest has the following columns:
- `talk_id`: Unique identifier for the talk
- `title`: Title of the talk/video
- `text`: Complete transcript of the talk
- `label`: Humor label (1 for humorous, 0 for non-humorous)

## Statistics

- Training set: 842 talks (all labeled as humorous)
- Validation set: 45 talks (all labeled as humorous)

## Usage

### Building the Dataset

To rebuild the talk-level manifests from scratch:

```bash
python build_talk_level_humor_manifests.py
```

### Training a Model

Use the provided script to train a DistilBERT model on the talk-level dataset:

```bash
bash train_distilbert_talk_level_ec2.sh
```

This script:
1. Uploads necessary files to the EC2 instance
2. Runs the training with optimized hyperparameters for talk-level data:
   - Increased max sequence length (256 vs 128)
   - More epochs (15 vs 10)
   - Smaller batch size (16 vs 32)
   - Higher learning rate (3e-5 vs 2e-5)
3. Downloads the trained model back to the local machine

## Performance

The talk-level approach offers several advantages over utterance-level:
1. **Contextual Understanding**: Models can understand humor in its full context
2. **Reduced Ambiguity**: Clear labels per talk rather than per utterance
3. **Simpler Training**: Fewer but more meaningful examples

## Future Work

Potential improvements:
1. Obtain additional non-humorous talks to balance the dataset
2. Experiment with larger models (BERT, RoBERTa) that can handle longer sequences
3. Implement techniques to extract key humorous segments from longer talks
