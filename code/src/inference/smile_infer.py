import torch
import argparse
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.models import ResNet18_Weights
import pytorch_lightning as pl
import sys

# Add project root to sys.path to allow importing train_smile
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Assuming train_smile.py is in the project root or accessible in PYTHONPATH
    from train_smile import SmileClassifier
except ImportError:
    print("Error: Could not import SmileClassifier from train_smile.py.")
    print("Ensure train_smile.py is in the project root or your PYTHONPATH.")
    sys.exit(1)


def infer_smile(image_path, checkpoint_path):
    """
    Loads a SmileClassifier checkpoint and predicts the smile probability for a single image.

    Args:
        image_path (str): Path to the input image file.
        checkpoint_path (str): Path to the trained model checkpoint (.ckpt).

    Returns:
        float: Probability of the image containing a smile, or None if an error occurs.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    try:
        # Load model from checkpoint
        model = SmileClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()  # Set to evaluation mode
        model.freeze() # Freeze weights

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Loaded model from {checkpoint_path} onto {device}.")

        # Load and preprocess the image
        image_raw = read_image(image_path, mode=ImageReadMode.RGB).to(device) # Load as CxHxW tensor to device

        # Get the preprocessing transforms from the loaded model or weights
        # Using the transforms associated with the weights the model was trained with
        weights = ResNet18_Weights.IMAGENET1K_V1 # Assuming trained with these weights
        preprocess = weights.transforms()
        
        # Apply transforms - Note: transforms expect input on CPU usually, adjust if needed
        # Or apply transforms before moving to device if they require CPU tensors/PIL images
        # Let's assume preprocess handles device appropriately or works on PIL images first
        
        # Convert tensor to PIL for transforms that require it
        from torchvision import transforms as T
        temp_transform = T.Compose([T.ToPILImage(), preprocess])
        image_processed = temp_transform(image_raw.cpu()).unsqueeze(0).to(device) # Add batch dim and move back to device


        print(f"Input image shape after processing: {image_processed.shape}") # Should be [1, 3, H, W]

        # Perform inference
        with torch.no_grad():
            logit = model(image_processed).squeeze() # Get single logit output
            prob = torch.sigmoid(logit) # Convert logit to probability

        smile_probability = prob.item()
        print(f"Predicted smile probability: {smile_probability:.4f}")
        return smile_probability

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smile Inference Script")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')

    args = parser.parse_args()

    probability = infer_smile(args.image_path, args.ckpt)

    if probability is not None:
        print(f"\nFinal Result -> Smile Probability for {os.path.basename(args.image_path)}: {probability:.4f}")
    else:
        print("\nInference failed.")
