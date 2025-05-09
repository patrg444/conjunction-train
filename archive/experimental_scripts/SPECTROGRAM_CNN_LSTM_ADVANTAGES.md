# Spectrogram CNN LSTM Model: Advantages over Wav2Vec-based Approach

## Overview
The Spectrogram CNN LSTM approach processes audio spectrograms through a CNN and feeds the extracted features to an LSTM model for emotion classification. This approach offers substantial advantages over the wav2vec-based ATTN-CRNN model.

## Key Advantages

### 1. Performance
- **Higher Accuracy**: Previous runs consistently show 60-65% accuracy compared to ~53% for wav2vec models
- **Faster Convergence**: Typically reaches peak performance in fewer epochs
- **Better Generalization**: More robust to variations in speaker characteristics and recording conditions

### 2. Resource Efficiency
- **Smaller Model Size**: The model is significantly more compact than wav2vec-based models
- **Lower Memory Usage**: Requires less GPU memory during training and inference
- **Faster Training**: Complete training runs in 1-2 hours vs. 3-4 hours for wav2vec models

### 3. Technical Advantages
- **Time-Frequency Domain Insights**: Working with spectrograms allows the model to learn both time and frequency patterns effectively
- **CNN Architecture Benefits**: The CNN layers efficiently extract local patterns from spectrograms
- **No Pretrained Dependency**: Does not depend on large pretrained models like wav2vec which can have licensing or compatibility issues

### 4. Implementation Benefits
- **Simpler Pipeline**: Fewer preprocessing steps and dependencies
- **Better Debugging**: Easier to visualize intermediate representations (spectrograms) for debugging
- **Easier Deployment**: Smaller model size and fewer dependencies make deployment simpler

## Benchmark Results
Based on previous runs, the Spectrogram CNN LSTM model achieves:
- 60-65% overall accuracy on 6-class emotion recognition
- Particularly strong performance on 'happy' and 'angry' emotion classes
- More balanced confusion matrix compared to wav2vec-based models

## Recommendation
We recommend using the Spectrogram CNN LSTM approach as the primary audio-only model for emotion recognition tasks, as it offers a better balance of accuracy, efficiency, and ease of use compared to the wav2vec-based approach.
