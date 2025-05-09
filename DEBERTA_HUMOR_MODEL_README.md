# DeBERTa-v3 Humor Detection Model

This document describes the DeBERTa-v3 model being trained for humor detection and the process for monitoring and deploying it to EC2.

## Model Architecture

- **Base Model**: microsoft/deberta-v3-base
- **Task**: Binary classification of text for humor detection
- **Output Layer**: Linear classification head with 2 output neurons
- **Tokenizer**: DeBERTa tokenizer from the pretrained model

## Training Configuration

- **Max Sequence Length**: 128
- **Batch Size**: 16
- **Learning Rate**: 2.0e-05
- **Epochs**: 1
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Dropout**: 0.2
- **LR Scheduler**: Cosine with warmup
- **Gradient Clipping**: 1.0
- **FP16 Training**: Enabled for faster computation

## Dataset

The model is being trained on the UR-Funny dataset of humor text, specifically using:
- **Training Dataset**: ur_funny_train_humor_cleaned.csv
- **Validation Dataset**: ur_funny_val_humor_cleaned.csv

## Monitoring Process

The training process is monitored using the `monitor_deberta_training.sh` script, which:

1. Checks if the training process (PID 14561) is still running
2. Looks for checkpoint files in the training directory
3. Creates a summary of the training when completed
4. Automatically uploads the trained model to EC2 using `upload_deberta_files.sh`

To start the monitoring process in a separate terminal:

```bash
./monitor_deberta_training.sh
```

## Deployment

When training completes, the model will be automatically uploaded to the EC2 instance using the `upload_deberta_files.sh` script. The upload process:

1. Creates a timestamped directory on the EC2 instance
2. Transfers all model files with their directory structure
3. Optionally includes the manifest files for the datasets
4. Creates a README with usage instructions on the remote server

To manually upload the model files:

```bash
./upload_deberta_files.sh [--include_manifests] [--include_source]
```

## Usage Example

Once deployed, the model can be used for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_path = "/path/to/model"  # Replace with actual path on EC2
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Run inference
text = "Your text to classify"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
prediction = torch.argmax(probabilities, dim=-1).item()
confidence = probabilities[0][prediction].item()

print(f"Prediction: {'Humorous' if prediction == 1 else 'Not humorous'}")
print(f"Confidence: {confidence:.2f}")
```

## Performance

Performance metrics will be updated once training completes. Expected metrics include:
- Accuracy
- F1 Score
- Precision
- Recall
- Confusion Matrix

## Notes

- The DeBERTa-v3 model is currently considered state-of-the-art for many NLP tasks and should provide strong performance for humor detection.
- Future work could include fine-tuning with different learning rates, using larger batch sizes on GPU instances, or combining with other humor datasets.
- The model can be integrated with the fusion model for multimodal humor detection combining text, audio, and visual features.
