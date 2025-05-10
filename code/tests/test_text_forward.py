import torch
import pytest
import sys
import os
from transformers import AutoTokenizer

# Add project root to sys.path to allow importing train_distil_humor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from train_distil_humor import DistilHumorClassifier
except ImportError:
    pytest.skip("Could not import DistilHumorClassifier, skipping test.", allow_module_level=True)

@pytest.mark.skipif(DistilHumorClassifier is None, reason="DistilHumorClassifier not available")
def test_text_humor_forward_pass():
    """
    Tests the forward pass of the DistilHumorClassifier model with dummy input.
    """
    # Model parameters
    model_name = 'distilbert-base-uncased'
    num_classes = 2 # Humor/Non-Humor
    # Dummy scheduler params needed for init if they are hyperparameters
    model = DistilHumorClassifier(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=2e-5,
        warmup_steps=10, # Dummy value
        total_steps=100 # Dummy value
    )

    # Dummy input
    batch_size = 2
    seq_length = 64 # Example sequence length
    dummy_input_ids = torch.randint(0, model.bert.config.vocab_size, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    # Perform forward pass
    try:
        with torch.no_grad():
            model.eval()
            output = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)

        # Check output shape (Batch size, num_classes)
        assert output.shape == (batch_size, num_classes), f"Output shape mismatch: Expected ({batch_size}, {num_classes}), Got {output.shape}"
        print(f"Text Humor forward pass successful. Output shape: {output.shape}")

    except Exception as e:
        pytest.fail(f"Text Humor forward pass failed with error: {e}")

# To run this test: pytest tests/test_text_forward.py
