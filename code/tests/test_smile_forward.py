import torch
import pytest
import sys
import os

# Add project root to sys.path to allow importing train_smile
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from train_smile import SmileClassifier
except ImportError:
    pytest.skip("Could not import SmileClassifier, skipping test.", allow_module_level=True)

@pytest.mark.skipif(SmileClassifier is None, reason="SmileClassifier not available")
def test_smile_forward_pass():
    """
    Tests the forward pass of the SmileClassifier model with dummy input.
    """
    # Model parameters (assuming binary classification)
    num_classes = 1 # Output is single logit for BCEWithLogitsLoss
    model = SmileClassifier(learning_rate=1e-4) # LR doesn't matter for forward pass

    # Dummy input (Batch size 2, 3 channels, 112x112 image - typical input size for ResNet)
    # Use the transform size defined in the model or standard ImageNet size
    img_size = 224 # Standard ImageNet size, adjust if model expects different
    if hasattr(model, 'preprocess') and hasattr(model.preprocess, 'resize_size'):
         # Attempt to get size from model's transforms if available
         # Note: Accessing internal transform details might be fragile
         try:
              img_size = model.preprocess.resize_size[0] # Often a list/tuple
         except:
              print("Could not infer image size from model transforms, using default 224.")
              img_size = 224


    dummy_input = torch.randn(2, 3, img_size, img_size)

    # Perform forward pass
    try:
        with torch.no_grad():
            model.eval()
            output = model(dummy_input)

        # Check output shape (Batch size, num_classes=1 for BCE)
        assert output.shape == (2, num_classes), f"Output shape mismatch: Expected (2, {num_classes}), Got {output.shape}"
        print(f"Smile forward pass successful. Output shape: {output.shape}")

    except Exception as e:
        pytest.fail(f"Smile forward pass failed with error: {e}")

# To run this test: pytest tests/test_smile_forward.py
