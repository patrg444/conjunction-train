# Wav2Vec Emotion Recognition: Neutral-Calm Mapping

## Overview

This document details the implementation of mapping the 'calm' emotion category to the 'neutral' category in our Wav2Vec audio emotion recognition model. This approach addresses several challenges in speech emotion recognition and improves model performance.

## Motivation

When working with speech emotion datasets, we observed several issues with the 'calm' and 'neutral' emotion categories:

1. **Ambiguous Boundaries**: The distinction between 'calm' and 'neutral' emotions is often subjective and inconsistent across datasets and human annotators.

2. **Limited Samples**: The 'calm' category typically has significantly fewer samples than other emotion categories (as seen in the class distribution where 'calm' had only 173 samples compared to over 1,300 for most other emotions).

3. **Class Imbalance**: The severe imbalance necessitates artificially high class weighting (6.45x for 'calm' vs ~0.85x for most other classes).

4. **Perceptual Similarity**: From a perceptual standpoint, 'calm' and 'neutral' speech share many acoustic properties compared to more distinct emotions like 'angry' or 'happy'.

## Implementation

We modified the emotion mapping in the data loading pipeline:

```python
# Original emotion mapping
emotion_to_index = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    ...
}

# Modified emotion mapping that combines 'calm' with 'neutral'
emotion_to_index = {
    'neutral': 0,
    'calm': 0,  # Now mapped to the same index as 'neutral'
    'happy': 2,
    ...
}
```

The full implementation can be found in `fixed_v5_script_neutral_calm.py`, which:

1. Maps 'calm' samples to the same class index as 'neutral'
2. Adjusts class weights automatically based on the combined samples
3. Maintains the original model architecture 
4. Preserves compatibility with the original deployment pipeline

## Expected Benefits

1. **Increased Training Data**: The neutral class now contains both 'neutral' and 'calm' samples, increasing its representation.

2. **More Balanced Classes**: Reducing the number of emotion categories creates a more balanced dataset.

3. **Reduced Confusion**: Eliminating the ambiguity between 'calm' and 'neutral' allows the model to focus on learning more distinct emotion boundaries.

4. **Improved Accuracy**: Early results show improved validation accuracy compared to the original model.

5. **Simpler Deployment**: The model output is more straightforward with fewer emotion categories to consider.

## Usage

To utilize this approach:

1. Deploy the training script:
   ```
   ./deploy_neutral_calm_wav2vec_training.sh
   ```

2. Monitor training progress:
   ```
   ./monitor_neutral_calm_training.sh
   ```

3. Download the trained model:
   ```
   ./download_neutral_calm_model.sh
   ```

## Evaluation

The model's performance should be evaluated based on:

1. Overall validation accuracy
2. Per-class precision and recall
3. Confusion matrix to verify reduced confusion between emotion classes
4. Practical performance in real-world speech emotion recognition tasks

## Future Work

1. Explore further emotion category consolidation where appropriate
2. Test the approach on additional datasets
3. Conduct user studies to verify the practical impact of the combined categories
4. Integrate the model into the real-time emotion recognition pipeline

## References

- Kerkeni, L., et al. (2019). "Automatic Speech Emotion Recognition Using Machine Learning."
- Akçay, M.B., Oğuz, K. (2020). "Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers."
