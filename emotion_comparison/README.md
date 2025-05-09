# Emotion Recognition Method Comparison Framework

A comprehensive framework for comparing different emotion recognition methods on video datasets.

## Features

- **Multiple Feature Extraction Methods:**
  - CNN3D: 3D convolutional features from video frames
  - FACS: Facial Action Coding System features
  - Multi-Region: Features from different facial regions
  - Pretrained: Features from pretrained models (FaceNet, VGG, etc.)

- **Multiple Classifier Support:**
  - RandomForest, SVM, Neural Networks, and more
  - Hyperparameter optimization with grid search

- **Visualization and Reporting:**
  - HTML reports with embedded visualizations
  - Confusion matrices
  - Classification reports
  - Cross-dataset analysis

- **Reproducibility:**
  - Docker support
  - CI/CD with GitHub Actions
  - Deterministic results

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-recognition-comparison.git
cd emotion-recognition-comparison

# Install dependencies
pip install -r emotion_comparison/requirements.txt
```

### Docker Installation (Recommended)

```bash
# Build the Docker image
cd emotion_comparison
docker build -t emotion-comparison .

# Run with mounted volumes for data and results
docker run -v /path/to/videos:/app/videos -v /path/to/results:/app/results emotion-comparison --help
```

## Usage

### Basic Usage

```bash
# Run a test on 10 videos
./run_emotion_comparison.sh --test

# Run on a small sample of 50 videos
./run_emotion_comparison.sh --sample

# Run full comparison on RAVDESS dataset
./run_emotion_comparison.sh --ravdess

# Run full comparison on CREMA-D dataset
./run_emotion_comparison.sh --cremad
```

### Advanced Options

```bash
# Run with hyperparameter grid search
./run_emotion_comparison.sh --test --grid-search

# Run without interactive prompts
./run_emotion_comparison.sh --ravdess --auto

# Generate visualizations for existing results
./run_emotion_comparison.sh --visualize

# View help
./run_emotion_comparison.sh --help
```

## Hyperparameter Grid Search

Enable hyperparameter optimization to find the best classifier configurations:

```bash
./run_emotion_comparison.sh --ravdess --grid-search
```

Customize the parameter grid in `param_grid_example.json`:

```json
{
  "random_forest": {
    "n_estimators": [50, 100, 200],
    "max_depth": [null, 10, 20, 30]
  },
  "svm": {
    "C": [0.1, 1.0, 10.0],
    "kernel": ["linear", "rbf", "poly"]
  }
}
```

## Directory Structure

```
emotion_comparison/
├── Dockerfile              # Docker configuration
├── run_comparison.py       # Main comparison script
├── hyperparam_search.py    # Grid search functionality
├── param_grid_example.json # Example parameter grid
├── common/                 # Shared utilities
│   ├── visualization.py    # Visualization tools
│   ├── evaluation.py       # Evaluation metrics
│   └── dataset_utils.py    # Dataset handling
├── cnn3d/                  # CNN3D method
├── facs/                   # FACS method
├── multi_region/           # Multi-region method
└── pretrained/             # Pretrained models method
```

## Extending the Framework

### Adding a New Feature Extraction Method

1. Create a new directory for your method in `emotion_comparison/`
2. Implement a `feature_extractor.py` with an `extract_features(video_path, output_dir)` function
3. Update `run_comparison.py` to include your method

### Adding a New Classifier

Modify `hyperparam_search.py` to add your classifier to the `classifier_map` dictionary:

```python
classifier_map = {
    'random_forest': RandomForestClassifier,
    'svm': SVC,
    # Add your classifier here
    'your_classifier': YourClassifier,
}
```

## Docker Usage

The included Dockerfile creates an isolated environment with all dependencies:

```bash
# Build the Docker image
docker build -t emotion-comparison .

# Run a test using Docker
docker run emotion-comparison --test

# Mount volumes for data access
docker run -v /path/to/videos:/app/videos -v /path/to/results:/app/results \
    emotion-comparison --ravdess
```

## CI/CD Integration

The framework includes GitHub Actions configuration for continuous integration:

- Automatically runs tests on pull requests
- Validates that changes don't break existing functionality
- Ensures consistent behavior across environments

## License

This project is licensed under the MIT License - see the LICENSE file for details.
