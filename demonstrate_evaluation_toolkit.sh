#!/bin/bash
# This script demonstrates the WAV2VEC evaluation toolkit features
# It runs quick tests of each component with minimal data

echo "===== WAV2VEC Emotion Recognition Evaluation Toolkit Demo ====="
echo ""
echo "This demonstration will show how to use the toolkit components with sample data."
echo "For full evaluation with all data, please use the individual scripts directly."
echo ""

# Create directories
mkdir -p demo_results/labels
mkdir -p demo_results/evaluation
mkdir -p demo_results/cross_validation

# Step 1: Verify emotion labels on a limited set
echo "Step 1: Verifying emotion labels (limited test)"
echo "---------------------------------------------"
# This will run with a depth limit to find just a few files for demo purposes
./verify_emotion_labels.py --output-dir demo_results/labels

echo ""
echo "Step 2: Running local model evaluation (limited test)"
echo "---------------------------------------------------"
# Run with reduced search and just basic evaluation (no cross-validation)
./run_wav2vec_full_dataset_evaluation.sh --output-dir demo_results/evaluation

echo ""
echo "Step 3: Exploring cross-validation options"
echo "----------------------------------------"
# Just print the help message for demonstration
echo "The deploy_cross_validation.sh script can perform full cross-validation on the server."
echo "For this demo, we'll just show the script parameters:"
echo ""
echo "Usage: ./deploy_cross_validation.sh"
echo "This script uploads the evaluation tools to the EC2 server and runs cross-validation"
echo "Results are downloaded to results/server_cross_validation/"
echo ""

echo "Step 4: Reviewing evaluation results"
echo "-----------------------------------"
echo "The toolkit generates the following results:"
echo "* Confusion matrix visualization"
echo "* Per-class accuracy, F1, and support metrics"
echo "* Overall accuracy, balanced accuracy, and macro F1 score"
echo "* Distribution of emotion labels across the dataset"
echo ""

# Check if any results were generated
if [ -d "demo_results/labels" ] && [ "$(ls -A demo_results/labels)" ]; then
    echo "Demo results were saved to:"
    echo "* Label analysis: demo_results/labels/"
    ls -la demo_results/labels/
fi

if [ -d "demo_results/evaluation" ] && [ "$(ls -A demo_results/evaluation)" ]; then
    echo "* Evaluation results: demo_results/evaluation/"
    ls -la demo_results/evaluation/
fi

echo ""
echo "For a real evaluation, you would run:"
echo "1. ./verify_emotion_labels.py - To analyze label distribution"
echo "2. ./run_wav2vec_full_dataset_evaluation.sh - For local evaluation"
echo "3. ./deploy_cross_validation.sh - For server-side cross-validation"
echo ""
echo "See WAV2VEC_EVALUATION_README.md for detailed documentation."
echo ""
echo "===== Demo Complete ====="
