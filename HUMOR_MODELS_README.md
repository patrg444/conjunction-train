# Humor Detection Models

This repository contains tools for training and deploying state-of-the-art transformer models for humor detection. Currently, we support two powerful models:

1. **DeBERTa-v3-base** - Running locally with real-time monitoring
2. **XLM-RoBERTa-large** - Configured for G5.4xlarge EC2 instance with 24GB GPU memory

## Model Comparison

| Feature | DeBERTa-v3-base | XLM-RoBERTa-large |
|---------|----------------|-------------------|
| Parameters | 184M | 559M |
| Language Support | English | Multilingual (100 languages) |
| Training Environment | Local / CPU or GPU | EC2 G5.4xlarge (24GB GPU) |
| Batch Size | 16 | 8 |
| Learning Rate | 2e-5 | 1e-5 |
| Training Speed | Moderate | Fast (with GPU acceleration) |
| Memory Requirements | Lower | Higher |

## Current Setup

### Local Training (DeBERTa-v3)

The DeBERTa-v3-base model is currently being trained locally with real-time monitoring. This model has excellent performance on English text classification tasks and is more efficient in terms of memory usage.

To monitor the current training process:
```bash
./monitor_deberta_training.sh
```

This will provide real-time updates including:
- Log streaming of training output
- Checkpoint detection
- Automatic upload to EC2 when completed

### EC2 GPU Training (XLM-RoBERTa)

For more computational power, the XLM-RoBERTa-large model is configured to run on an AWS G5.4xlarge instance with 24GB of GPU memory. This model is larger and can handle multilingual inputs.

To deploy and train on the EC2 instance:
```bash
./train_xlm_roberta_large_ec2.sh
```

This will:
1. Transfer necessary manifests to the EC2 instance
2. Create a Python training script on the EC2 instance
3. Install required dependencies
4. Start training with mixed precision (FP16)
5. Save the model upon completion

## Training Data

Both models use the UR-Funny dataset for humor detection training, with the data split into:
- `ur_funny_train_humor_cleaned.csv` - Training data
- `ur_funny_val_humor_cleaned.csv` - Validation data

## Model Deployment

Once training is complete, models are automatically uploaded to the EC2 instance for deployment and inference. Access them via:

```bash
# For DeBERTa (uploaded automatically)
ssh -i "/Users/patrickgloria/Downloads/gpu-key.pem" ubuntu@$(cat aws_instance_ip.txt) "ls -la ~/humor_models/"

# For XLM-RoBERTa (saved on EC2)
ssh -i "/Users/patrickgloria/Downloads/gpu-key.pem" ubuntu@$(cat aws_instance_ip.txt) "ls -la ~/training_logs_humor/xlm-roberta-large_g5/"
```

## Usage Example

After training, you can use either model for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Choose model path based on which model you want to use
# model_path = "path/to/deberta-v3-base"  # Local DeBERTa model
model_path = "/home/ubuntu/training_logs_humor/xlm-roberta-large_g5/final_model"  # EC2 XLM-RoBERTa

# Load the appropriate tokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")  # For DeBERTa
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")  # For XLM-RoBERTa

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Run inference
text = "Why did the chicken cross the road? To get to the other side!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
prediction = torch.argmax(probabilities, dim=-1).item()
confidence = probabilities[0][prediction].item()

print(f"Prediction: {'Humorous' if prediction == 1 else 'Not humorous'}")
print(f"Confidence: {confidence:.2f}")
```

## Future Improvements

1. Fine-tuning with gradient accumulation to enable larger effective batch sizes
2. Model distillation to create smaller, deployment-friendly models 
3. Ensemble methods combining both models for improved accuracy
4. Cross-lingual humor detection leveraging XLM-RoBERTa's multilingual capabilities
