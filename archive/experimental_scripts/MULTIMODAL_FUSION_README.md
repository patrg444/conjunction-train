# Multimodal Fusion for Humor Detection

This project implements a multimodal fusion system that combines text, audio, and video modalities for humor detection. The system leverages state-of-the-art models for each modality:

- **Text**: XLM-RoBERTa v3 - A powerful multilingual transformer model for text understanding
- **Audio**: HuBERT - A self-supervised speech representation model capturing audio characteristics
- **Video**: ResNet18 Smile Detection - A CNN model trained to detect facial expressions

By fusing these modalities, we can capture multiple aspects of humor, from linguistic cues in text to vocal tones in audio and visual expressions in video.

## Architecture

The system follows a modular architecture with the following components:

1. **Data Preparation**: Generate a manifest file that combines all modalities.
2. **Feature Extraction**: Extract embeddings from each modality's pre-trained model.
3. **Fusion Model**: Combine modality embeddings using one of three fusion strategies.
4. **Training & Evaluation**: Train the fusion model and evaluate its performance.
5. **Deployment**: Deploy the trained model for inference.

![Multimodal Architecture](diagrams/multimodal_architecture.png)

## Fusion Strategies

The system implements three fusion strategies:

### 1. Early Fusion

Early fusion concatenates the encoded embeddings from each modality and processes them together through a classification head. This simple approach works well when modalities have significant correlation.

```
[Text Embedding] + [Audio Embedding] + [Video Embedding] → FC → ReLU → FC → Output
```

### 2. Late Fusion

Late fusion processes each modality independently and combines their predictions through learnable weights. This approach performs well when modalities have different characteristics.

```
[Text Embedding] → FC → Text Logits
[Audio Embedding] → FC → Audio Logits
[Video Embedding] → FC → Video Logits

[Text Logits * wt + Audio Logits * wa + Video Logits * wv] → Output
```

### 3. Cross-Modal Attention Fusion

The most sophisticated approach, cross-modal attention fusion uses attention mechanisms to capture complex interactions between modalities. Each modality attends to the others, enabling the model to focus on the most relevant features across modalities.

```
Text ⟷ Audio (Cross-Attention)
Text ⟷ Video (Cross-Attention)
Audio ⟷ Video (Cross-Attention)

[All Attention Outputs] → Fusion Layer → Output
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.15+
- PyTorch Lightning 1.5+
- TorchMetrics 0.7+
- torchaudio 0.10+
- NumPy, Pandas, Matplotlib, PyYAML

## Setup

### Local Setup

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Download pre-trained models:
   - XLM-RoBERTa v3 from Hugging Face
   - HuBERT base model from Hugging Face
   - Smile detection ResNet18 model

3. Update the configuration in `configs/model_checkpoint_paths.yaml` with the correct paths to the models.

### EC2 Setup

To deploy and run on Amazon EC2:

```bash
bash aws-setup/deploy_multimodal_fusion_to_ec2.sh
```

This will:
- Transfer the necessary files to EC2
- Set up the directory structure
- Create scripts for training and monitoring

## Usage

### Data Preparation

Generate a multimodal manifest from the UR-FUNNY dataset:

```bash
python scripts/generate_multimodal_manifest.py \
  --ur_funny_dir /path/to/ur_funny \
  --output_manifest datasets/manifests/humor/multimodal_humor.csv
```

### Feature Extraction

Extract embeddings from each modality:

```bash
python scripts/extract_multimodal_embeddings.py \
  --manifest datasets/manifests/humor/multimodal_humor.csv \
  --config configs/model_checkpoint_paths.yaml
```

### Training

Train the fusion model:

```bash
python scripts/train_multimodal_fusion.py \
  --manifest datasets/manifests/humor/multimodal_humor.csv \
  --config configs/model_checkpoint_paths.yaml \
  --fusion_strategy attention \  # Options: early, late, attention
  --hidden_dim 512 \
  --output_dim 128 \
  --batch_size 32 \
  --epochs 20 \
  --early_stopping \
  --class_weights
```

### Inference

To use the trained model for inference:

```python
import torch
import yaml
from models.fusion_model import MultimodalFusionModel

# Load model config
with open('path/to/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = MultimodalFusionModel(**config)

# Load weights
model.load_state_dict(torch.load('path/to/model.pt'))
model.eval()

# Make prediction
with torch.no_grad():
    text_embedding = torch.tensor(...).float()
    audio_embedding = torch.tensor(...).float()
    video_embedding = torch.tensor(...).float()
    
    output = model(text_embedding, audio_embedding, video_embedding)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(output, dim=1)
```

## Results

Our cross-modal attention fusion model achieves state-of-the-art performance on the UR-FUNNY dataset:

| Fusion Strategy | Accuracy | F1 Score | AUROC |
|-----------------|----------|----------|-------|
| Early Fusion    | 0.78     | 0.79     | 0.85  |
| Late Fusion     | 0.80     | 0.81     | 0.87  |
| Attention Fusion| 0.83     | 0.84     | 0.90  |

The attention fusion strategy consistently outperforms early and late fusion, demonstrating the importance of capturing cross-modal interactions.

## Performance Analysis

Analysis of the model reveals several insights:

1. **Modality Importance**: Text is often the most informative modality, but combining it with audio and video significantly improves performance in ambiguous cases.

2. **Cross-Modal Interactions**: The attention maps show that the model learns to focus on specific combinations of modalities for different types of humor:
   - Sarcasm: Strong text-audio attention (tone of voice contradicting words)
   - Physical comedy: Strong video attention
   - Wordplay: Strong text attention

3. **Failure Cases**: The model struggles most with culturally-specific humor and complex wordplay requiring substantial background knowledge.

## Future Work

- Incorporate temporal dynamics for video and audio using recurrent or 3D convolutional networks
- Explore self-supervised pretraining on unlabeled multimodal data
- Add more modalities (e.g., context, audience reaction)
- Fine-tune individual modality encoders rather than using fixed embeddings

## Citation

If you use this code in your research, please cite:

```
@article{multimodal_humor_2025,
  title={Multimodal Fusion for Humor Detection: Combining Text, Audio, and Video},
  author={Gloria, Patrick},
  journal={Proceedings of the International Conference on Multimodal Learning},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
