import torch
import torch.nn as nn
from transformers import WavLMForSequenceClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WavLMClassifier(nn.Module):
    """
    WavLM model for sequence classification (e.g., humor detection).
    Wraps the Hugging Face WavLMForSequenceClassification model.
    """
    def __init__(self, model_name="microsoft/wavlm-base-plus", num_classes=2, freeze_feature_extractor=False):
        """
        Args:
            model_name (str): Name of the Hugging Face WavLM model.
            num_classes (int): Number of output classes for classification.
            freeze_feature_extractor (bool): Whether to freeze the weights of the WavLM feature extractor.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_feature_extractor = freeze_feature_extractor

        logging.info(f"Loading WavLM model for sequence classification '{model_name}'...")
        try:
            # Load WavLMForSequenceClassification directly
            # This model already includes a pooling layer and a classification head
            self.wavlm_classifier = WavLMForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes # Set the number of output classes
            )
            # Add alias for backward compatibility with scripts expecting model.wavlm
            self.wavlm = self.wavlm_classifier.wavlm
            logging.info(f"WavLMForSequenceClassification model '{model_name}' loaded successfully with {num_classes} labels.")

            if freeze_feature_extractor:
                logging.info("Freezing WavLM feature extractor.")
                # Freeze the feature extractor and feature projection layers
                self.wavlm_classifier.wavlm.feature_extractor._freeze_parameters()
                self.wavlm_classifier.wavlm.feature_projection._freeze_parameters()
                # Optionally freeze the encoder layers as well
                # for layer in self.wavlm_classifier.wavlm.encoder.layers:
                #     for param in layer.parameters():
                #         param.requires_grad = False
                # logging.info("Freezing WavLM encoder layers.")

        except Exception as e:
            logging.error(f"Failed to load WavLMForSequenceClassification model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load WavLMForSequenceClassification model '{model_name}': {e}")

    def forward(self, input_values, attention_mask=None):
        """
        Forward pass through the WavLMForSequenceClassification model.

        Args:
            input_values (torch.Tensor): Input audio tensor (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Attention mask tensor (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits for classification (batch_size, num_classes).
        """
        # Pass inputs directly to the Hugging Face model
        outputs = self.wavlm_classifier(
            input_values,
            attention_mask=attention_mask,
            return_dict=True # Ensure outputs are returned as a dictionary
        )

        # The logits are directly available in the output dictionary
        logits = outputs.logits # Shape: (batch_size, num_classes)

        return logits

# Example Usage (for testing this script directly)
if __name__ == '__main__':
    print("--- Testing WavLMClassifier ---")

    # Dummy input data (batch_size=4, sequence_length=16000*1=16000 for 1 second at 16kHz)
    # WavLM expects raw waveform as input_values
    dummy_input_values = torch.randn(4, 16000)
    # Create a dummy attention mask (e.g., first 10000 tokens are real, rest are padding)
    dummy_attention_mask = torch.ones(4, 16000, dtype=torch.long)
    dummy_attention_mask[1, 10000:] = 0 # Example padding for batch item 1
    dummy_attention_mask[3, 12000:] = 0 # Example padding for batch item 3


    try:
        # Instantiate the model
        # Using a smaller model for faster testing
        model = WavLMClassifier(model_name="facebook/wav2vec2-base-960h", num_classes=2)
        print("\nModel instantiated.")

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dummy_input_values = dummy_input_values.to(device)
        dummy_attention_mask = dummy_attention_mask.to(device)
        print(f"Using device: {device}")

        # Perform a forward pass
        print("\nPerforming forward pass...")
        with torch.no_grad():
            logits = model(dummy_input_values, attention_mask=dummy_attention_mask)

        print("Forward pass successful.")
        print("Output logits shape:", logits.shape) # Should be [batch_size, num_classes] -> [4, 2]

        # Test with freezing feature extractor
        print("\nTesting with frozen feature extractor...")
        model_frozen = WavLMClassifier(model_name="facebook/wav2vec2-base-960h", num_classes=2, freeze_feature_extractor=True)
        model_frozen.to(device)
        print("Model with frozen feature extractor instantiated.")

        # Check if parameters are frozen
        frozen_params = [name for name, param in model_frozen.named_parameters() if not param.requires_grad]
        unfrozen_params = [name for name, param in model_frozen.named_parameters() if param.requires_grad]
        print(f"Number of frozen parameters: {len(frozen_params)}")
        print(f"Number of unfrozen parameters: {len(unfrozen_params)}")
        # print("Unfrozen parameters:", unfrozen_params) # Should primarily be the classifier weights/bias

        with torch.no_grad():
            logits_frozen = model_frozen(dummy_input_values, attention_mask=dummy_attention_mask)
        print("Forward pass with frozen model successful.")
        print("Output logits shape:", logits_frozen.shape)


    except Exception as e:
        print(f"\nError during model test: {e}")
        logging.error("Model test failed", exc_info=True)

    print("\n--- WavLMClassifier Test Finished ---")
