#!/bin/bash
# train_humor_fusion_workflow.sh
# Complete workflow script for training the Humor Fusion model
# This includes Smile branch, Text-Humor branch, and Fusion model training

set -e  # Exit immediately if a command exits with a non-zero status

echo "========================================="
echo "Starting Humor Fusion Training Workflow"
echo "========================================="

# Step 1: Train the Smile branch
echo "Step 1: Training Smile branch"
echo "----------------------------------------"
if [ ! -f "checkpoints/smile_best.ckpt" ]; then
    echo "Running Smile branch training..."
    bash shell/train_smile.sh
    
    if [ ! -f "checkpoints/smile_best.ckpt" ]; then
        echo "Error: Smile branch training failed to produce checkpoint!"
        exit 1
    fi
    echo "Smile branch checkpoint created successfully at checkpoints/smile_best.ckpt"
else
    echo "Smile branch checkpoint already exists. Skipping training."
fi
echo "Smile branch training complete."
echo ""

# Step 2: Train the Text-Humor branch
echo "Step 2: Training Text-Humor branch"
echo "----------------------------------------"
# Remove existing checkpoint if it exists
if [ -f "checkpoints/text_best.ckpt" ]; then
    echo "Removing existing Text-Humor checkpoint for retraining..."
    rm checkpoints/text_best.ckpt
fi

echo "Running Text-Humor branch training..."
bash enhanced_train_distil_humor.sh

if [ ! -f "checkpoints/text_best.ckpt" ]; then
    echo "Error: Text-Humor branch training failed to produce checkpoint!"
    exit 1
fi
echo "Text-Humor branch checkpoint created successfully at checkpoints/text_best.ckpt"
echo "Text-Humor branch training complete."
echo ""

# Step 3: Verify checkpoint paths in config
echo "Step 3: Verifying fusion model configuration"
echo "----------------------------------------"
config_file="configs/train_humor.yaml"
echo "Checking configuration file: $config_file"

# Define the expected paths for the checkpoints
correct_hubert="/home/ubuntu/conjunction-train/checkpoints/laughter_best.ckpt"
correct_smile="/home/ubuntu/conjunction-train/checkpoints/smile_best.ckpt"
correct_text="/home/ubuntu/conjunction-train/checkpoints/text_best.ckpt"

# Extract configuration from the file
hubert_config=$(grep "hubert_checkpoint:" "$config_file")
smile_config=$(grep "smile_checkpoint:" "$config_file")
text_config=$(grep "text_checkpoint:" "$config_file")

echo "Current configuration:"
echo "$hubert_config"
echo "$smile_config"
echo "$text_config"

# Check if paths are correctly set
hubert_check=$(echo "$hubert_config" | grep -c "$correct_hubert")
smile_check=$(echo "$smile_config" | grep -c "$correct_smile")
text_check=$(echo "$text_config" | grep -c "$correct_text")

if [ "$hubert_check" -ne 1 ] || [ "$smile_check" -ne 1 ] || [ "$text_check" -ne 1 ]; then
    echo "Error: Config file does not contain correct checkpoint paths!"
    echo "Should contain:"
    echo "hubert_checkpoint: $correct_hubert"
    echo "smile_checkpoint: $correct_smile"
    echo "text_checkpoint: $correct_text"
    exit 1
fi
echo "Configuration verified: checkpoint paths are correct."
echo ""

# Step 4: Train the Fusion model
echo "Step 4: Training Fusion model"
echo "----------------------------------------"
if [ ! -f "checkpoints/fusion_best.ckpt" ]; then
    echo "Running Fusion model training..."
    bash shell/train_fusion_humor.sh
    
    if [ ! -f "checkpoints/fusion_best.ckpt" ]; then
        echo "Error: Fusion model training failed to produce checkpoint!"
        exit 1
    fi
    echo "Fusion model checkpoint created successfully at checkpoints/fusion_best.ckpt"
else
    echo "Fusion model checkpoint already exists. Skipping training."
fi
echo "Fusion model training complete."
echo ""

# Step 5: Verification
echo "Step 5: Verifying all checkpoints"
echo "----------------------------------------"
checkpoints=(
    "checkpoints/smile_best.ckpt"
    "checkpoints/text_best.ckpt"
    "checkpoints/fusion_best.ckpt"
)

missing=0
for ckpt in "${checkpoints[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "Error: Checkpoint $ckpt is missing!"
        missing=1
    else
        echo "✓ Checkpoint $ckpt exists."
    fi
done

if [ "$missing" -eq 1 ]; then
    echo "Some checkpoints are missing. Workflow did not complete successfully."
    exit 1
fi

echo "========================================="
echo "Humor Fusion Training Workflow Complete!"
echo "========================================="
echo "All checkpoints verified:"
echo "- Smile branch: checkpoints/smile_best.ckpt"
echo "- Text-Humor branch: checkpoints/text_best.ckpt"
echo "- Fusion model: checkpoints/fusion_best.ckpt"
echo ""
echo "The trained humor fusion model is ready for use."
