#!/bin/bash
# Run the complete multimodal fusion pipeline for humor detection
# This script automates data preparation, embedding extraction, and model training

set -e

# Configuration
UR_FUNNY_DIR="/path/to/ur_funny"  # Update this path!
MANIFEST_PATH="datasets/manifests/humor/multimodal_humor.csv"
CONFIG_PATH="configs/model_checkpoint_paths.yaml"
EMBEDDING_DIR="embeddings"
OUTPUT_DIR="training_logs_humor/multimodal_attention_fusion"
FUSION_STRATEGY="attention"  # Options: early, late, attention

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${GREEN}===================================================="
echo -e "      Multimodal Fusion Pipeline for Humor Detection"
echo -e "====================================================${NC}"

# Check if UR-FUNNY directory exists
if [ ! -d "$UR_FUNNY_DIR" ]; then
  echo -e "${RED}Error: UR-FUNNY directory not found at $UR_FUNNY_DIR${NC}"
  echo -e "Please update the UR_FUNNY_DIR variable in this script with the correct path."
  exit 1
fi

# Create directories
mkdir -p embeddings/{text,audio,video}
mkdir -p datasets/manifests/humor
mkdir -p $OUTPUT_DIR

# Step 1: Generate multimodal manifest
echo -e "${YELLOW}Step 1: Generating multimodal manifest...${NC}"
python scripts/generate_multimodal_manifest.py --ur_funny_dir $UR_FUNNY_DIR --output_manifest $MANIFEST_PATH

# Check if manifest was created
if [ ! -f "$MANIFEST_PATH" ]; then
  echo -e "${RED}Error: Failed to generate manifest file at $MANIFEST_PATH${NC}"
  exit 1
fi
echo -e "${GREEN}Manifest generated successfully at $MANIFEST_PATH${NC}"

# Step 2: Extract embeddings
echo -e "${YELLOW}Step 2: Extracting embeddings from all modalities...${NC}"
python scripts/extract_multimodal_embeddings.py --manifest $MANIFEST_PATH --config $CONFIG_PATH --embedding_dir $EMBEDDING_DIR

# Check for extracted embeddings
TEXT_EMBEDDINGS=$(find $EMBEDDING_DIR/text -name "*.npy" | wc -l)
AUDIO_EMBEDDINGS=$(find $EMBEDDING_DIR/audio -name "*.npy" | wc -l)
VIDEO_EMBEDDINGS=$(find $EMBEDDING_DIR/video -name "*.npy" | wc -l)

echo -e "Extracted embeddings:"
echo -e "  - Text: $TEXT_EMBEDDINGS"
echo -e "  - Audio: $AUDIO_EMBEDDINGS"
echo -e "  - Video: $VIDEO_EMBEDDINGS"

if [ $TEXT_EMBEDDINGS -eq 0 ] || [ $AUDIO_EMBEDDINGS -eq 0 ] || [ $VIDEO_EMBEDDINGS -eq 0 ]; then
  echo -e "${RED}Warning: Some modalities have no extracted embeddings. The model may not train properly.${NC}"
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Step 3: Train fusion model
echo -e "${YELLOW}Step 3: Training multimodal fusion model with $FUSION_STRATEGY fusion...${NC}"
python scripts/train_multimodal_fusion.py \
  --manifest $MANIFEST_PATH \
  --config $CONFIG_PATH \
  --embedding_dir $EMBEDDING_DIR \
  --fusion_strategy $FUSION_STRATEGY \
  --hidden_dim 512 \
  --output_dim 128 \
  --batch_size 32 \
  --epochs 20 \
  --early_stopping \
  --class_weights \
  --experiment_name multimodal_${FUSION_STRATEGY}_fusion \
  --version v1 \
  --gpus $(python -c "import torch; print(1 if torch.cuda.is_available() else 0)")

# Check if model was trained
FINAL_MODEL_DIR="${OUTPUT_DIR}/final_model"
if [ ! -d "$FINAL_MODEL_DIR" ]; then
  echo -e "${RED}Error: Training failed. Model not found at $FINAL_MODEL_DIR${NC}"
  exit 1
fi

echo -e "${GREEN}Training complete! Model saved to $FINAL_MODEL_DIR${NC}"

# Step 4: Show results
echo -e "${YELLOW}Step 4: Results summary${NC}"
echo -e "Classification report:"
cat ${OUTPUT_DIR}/classification_report.txt

echo -e "\nConfusion matrix saved to ${OUTPUT_DIR}/confusion_matrix.png"

if [ -f "$FINAL_MODEL_DIR/model.pt" ]; then
  echo -e "\n${GREEN}Multimodal fusion model successfully trained!${NC}"
  echo -e "Model is available at: $FINAL_MODEL_DIR/model.pt"
  echo -e "Model config is available at: $FINAL_MODEL_DIR/model_config.yaml"
fi

echo -e "\n${GREEN}To use the model for inference, see the example in MULTIMODAL_FUSION_README.md${NC}"
