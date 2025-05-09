# WAV2VEC Emotion Recognition Model Evaluation Toolkit

This toolkit enables comprehensive evaluation of the WAV2VEC-based emotion recognition model. It provides tools for assessing model performance on full datasets, verifying emotion label consistency, and performing cross-validation.

## Available Tools

### 1. `evaluate_wav2vec_full_dataset.py`

This script performs complete evaluation of the WAV2VEC model on the full dataset. It:
- Searches for all available WAV2VEC feature files (.npz)
- Loads and processes the features
- Evaluates the model's performance
- Generates confusion matrices and performance metrics
- Performs cross-validation (optional)

### 2. `verify_emotion_labels.py`

This utility analyzes emotion labels in WAV2VEC feature files. It:
- Identifies all emotion labels present in the dataset
- Analyzes their distribution and checks for consistency
- Detects label anomalies or potential errors
- Generates visualizations of label distribution
- Verifies that labels match the expected classes

### 3. `run_wav2vec_full_dataset_evaluation.sh`

A wrapper script that simplifies running the evaluation locally. It:
- Creates necessary directories
- Downloads the model if not available locally
- Checks for AWS connectivity
- Runs the evaluation with appropriate parameters
- Organizes results in a timestamped directory

### 4. `deploy_cross_validation.sh`

A script for running cross-validation on the EC2 server where all data is located. It:
- Uploads evaluation scripts to the server
- Runs cross-validation directly on the server
- Downloads results to the local machine

## Getting Started

### Local Evaluation

To evaluate the model on local or accessible files:

```bash
./run_wav2vec_full_dataset_evaluation.sh
```

Options:
- `--cross-validation`: Enable cross-validation
- `--n-folds N`: Set number of cross-validation folds (default: 5)
- `--max-length L`: Set maximum sequence length (default: 221)
- `--output-dir DIR`: Specify output directory
- `--search-dir DIR`: Add directory to search for feature files

Example with options:
```bash
./run_wav2vec_full_dataset_evaluation.sh --cross-validation --n-folds 3 --search-dir /data/features
```

### Server-Side Cross-Validation

To run cross-validation on the server where all data is located:

```bash
./deploy_cross_validation.sh
```

This will upload evaluation scripts to the server, run cross-validation directly on the EC2 instance, and download the results to your local machine.

### Verifying Emotion Labels

To analyze emotion labels and their distribution:

```bash
./verify_emotion_labels.py --output-dir results/label_analysis
```

You can also specify specific directories to search:
```bash
./verify_emotion_labels.py --search-dir /path/to/features --output-dir results/label_analysis
```

## Understanding Results

### Evaluation Metrics

The evaluation produces several key metrics:
- **Accuracy**: Overall percentage of correctly classified samples
- **Balanced Accuracy**: Accuracy that accounts for class imbalance
- **Macro F1 Score**: Average F1 score across all classes
- **Per-Class Metrics**: Precision, recall, and F1 score for each emotion class

### Visualizations

- **Confusion Matrix**: Shows which emotions are commonly confused
- **Label Distribution**: Bar chart showing the distribution of emotion classes

### Cross-Validation Results

Cross-validation results include:
- **Per-Fold Metrics**: Performance metrics for each fold
- **Overall Metrics**: Average and standard deviation across all folds
- **Cross-Validation Confusion Matrix**: Aggregated confusion matrix across all folds

## Technical Details

### Sequence Padding

Features are padded to a fixed length (default: 221) for model compatibility. This value was determined based on the 95th percentile of sequence lengths in the original training data.

### Label Mapping

The model uses specific emotion class mappings. The `verify_emotion_labels.py` script helps ensure your labels are compatible with the model's expected classes.

### Finding Feature Files

The evaluation scripts search multiple potential locations for feature files. You can specify additional directories using the `--search-dir` option.

## Troubleshooting

### Common Issues

- **"No feature files found"**: Check the search paths and ensure .npz files are present
- **"Error loading model"**: Ensure the model file exists at the expected location
- **"Unknown labels detected"**: Your feature files may use different emotion labels than the model expects
- **"Cannot connect to EC2 instance"**: Ensure the AWS instance IP is correct in aws_instance_ip.txt

### Debug Tips

- Run `verify_emotion_labels.py` first to check if your data is compatible
- Examine any errors carefully to identify issues with paths or file formats
- If cross-validation fails, try evaluating without cross-validation first

## Further Development

To enhance the evaluation toolkit:
- Add more comprehensive data augmentation for cross-validation
- Implement confusion matrix normalization options
- Add support for model ensembling
- Enable comparison between multiple model versions
