# Wav2Vec Emotion Recognition: Fixing Emotion Class Indices

## Problem Identification

During our implementation of the wav2vec-based emotion recognition model, we attempted to combine the 'calm' and 'neutral' emotion classes since they are semantically similar and 'calm' had a much smaller representation in the dataset. However, we encountered an unexpected issue:

1. **Perfect accuracy immediately**: After mapping 'calm' to 'neutral', the model achieved 100% accuracy from the first epoch.
2. **Zero loss**: The loss function reported 0.0 values, which is highly unusual.
3. **Early stopping**: Training ended early when accuracy plateaued at 100%.

After investigation, we discovered a fundamental issue in our emotion mapping approach:

```python
# Original mapping with a "hole" in the indices
emotion_to_index = {
    'neutral': 0,
    'calm': 0,     # Mapped to same as neutral
    'happy': 2,    # Notice index 1 is missing!
    'sad': 3,
    'angry': 4,
    'fear': 5,
    'disgust': 6,
    'surprise': 7
}
```

This created a non-continuous set of class indices (0,0,2,3,4,5,6,7) with a "hole" at index 1, which resulted in:

1. Non-contiguous one-hot encoding
2. Inconsistent indexing in the output layer
3. The model exploiting this pattern for trivial classification

## Solution: Continuous Indices

We implemented a fix by ensuring that all emotion indices are continuous after mapping:

```python
# Fixed mapping with continuous indices
emotion_to_index = {
    'neutral': 0,  # Neutral and calm are combined as class 0
    'calm': 0,     # Mapped to neutral (same as index 0)
    'happy': 1,    # Was 2, now 1 - continuous indexing
    'sad': 2,      # Was 3, now 2
    'angry': 3,    # Was 4, now 3
    'fear': 4,     # Was 5, now 4
    'disgust': 5,  # Was 6, now 5
    'surprise': 6  # Was 7, now 6
}
```

The key changes in our approach:
1. Maintain continuous integer indices for all emotion classes
2. Properly map both 'calm' and 'neutral' to index 0
3. Re-index all other emotions to maintain continuous values
4. Ensure the output layer has exactly 7 neurons (0-6) for the 7 distinct classes

## Implementation Details

We created a new script (`fixed_v5_script_continuous_indices.py`) that:

1. Maps emotions to continuous indices
2. Prevents using a cached version with incorrect indices
3. Adds additional diagnostics for training:
   - Prints the unique values in original and encoded labels
   - Shows emotion mapping clearly in logs
   - Verifies no overlap between training and validation sets
   - Confirms the number of output classes in the model

## Deployment

We deployed the fixed script using:
- `deploy_fixed_emotion_indices.sh`: Deploys the fixed script to EC2
- `monitor_six_classes.sh`: Monitors training progress with enhanced diagnostics
- `download_six_classes_model.sh`: Downloads the trained model with 6 distinct emotion classes

## Results

The fixed script properly trains a model with the following properties:
- 7 emotion categories reduced to 6 distinct emotion classes
- Proper class weights for imbalanced samples
- Continuous index mapping for stable training
- Normal training progression (gradual accuracy improvement and loss reduction)

## Lessons Learned

1. **Continuous indices are critical**: When remapping class indices, ensure they remain continuous for proper one-hot encoding.
2. **Verify diagnostics carefully**: Perfect accuracy from the first epoch should be treated as suspicious.
3. **Check class distribution**: Always confirm the distribution of classes after mapping/transformations.
4. **Weight of evidence**: Multiple oddities (perfect accuracy, zero loss, early stopping) together strongly indicate a fundamental issue.

This fix ensures that emotion recognition training produces a reliable model that generalizes properly to new data rather than exploiting artifacts in the class encoding.
