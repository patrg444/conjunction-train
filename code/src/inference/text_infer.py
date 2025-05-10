import torch
import argparse
import os
from transformers import AutoTokenizer
import pytorch_lightning as pl
import sys
import torch.nn.functional as F # For softmax

# Add project root to sys.path to allow importing train_distil_humor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Assuming train_distil_humor.py is in the project root or accessible in PYTHONPATH
    from train_distil_humor import DistilHumorClassifier
except ImportError:
    print("Error: Could not import DistilHumorClassifier from train_distil_humor.py.")
    print("Ensure train_distil_humor.py is in the project root or your PYTHONPATH.")
    sys.exit(1)


def infer_text_humor(text, checkpoint_path, model_name='distilbert-base-uncased', max_length=128):
    """
    Loads a DistilHumorClassifier checkpoint and predicts the humor probability for a single text string.

    Args:
        text (str): The input text string.
        checkpoint_path (str): Path to the trained model checkpoint (.ckpt).
        model_name (str): Name of the transformer model used during training.
        max_length (int): Max sequence length used during training.

    Returns:
        float: Probability of the text being humorous (class 1), or None if an error occurs.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    try:
        # Load model from checkpoint
        # We need to provide hparams if they weren't saved automatically or differ
        # Let's assume load_from_checkpoint works or adjust if needed
        model = DistilHumorClassifier.load_from_checkpoint(
            checkpoint_path,
            # Provide hparams if necessary, e.g., if they weren't saved in the checkpoint
            # model_name=model_name,
            # num_classes=2 # Assuming binary
        )
        model.eval()  # Set to evaluation mode
        model.freeze() # Freeze weights

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Loaded model from {checkpoint_path} onto {device}.")

        # Load tokenizer and preprocess text
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loaded tokenizer: {model_name}")

        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        print(f"Input text tokenized. Shape: {input_ids.shape}")

        # Perform inference
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)

        # Assuming class 1 is 'Humor'
        humor_probability = probs[0, 1].item() # Get probability of the positive class (index 1)

        print(f"Predicted humor probability: {humor_probability:.4f}")
        return humor_probability

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Humor Inference Script")
    parser.add_argument('--text', type=str, required=True, help='Input text string to classify.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Name of the transformer model used for training.')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length used during training.')


    args = parser.parse_args()

    probability = infer_text_humor(args.text, args.ckpt, args.model_name, args.max_length)

    if probability is not None:
        print(f"\nFinal Result -> Humor Probability for text: {probability:.4f}")
    else:
        print("\nInference failed.")
