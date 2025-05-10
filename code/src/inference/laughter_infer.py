import torch
import torchaudio
import argparse
import os
import sys
import yaml
from transformers import AutoFeatureExtractor

# Add project root to sys.path to allow importing train_fusion_humor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Assuming train_fusion_humor.py is in the 'scripts' directory relative to project root
    from scripts.train_fusion_humor import FusionModel
except ImportError as e:
    print(f"Error: Could not import FusionModel from scripts.train_fusion_humor: {e}")
    print("Ensure train_fusion_humor.py is in the 'scripts' directory and the project structure is correct.")
    sys.exit(1)

def load_yaml(file_path):
    """Load a YAML configuration file."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None

def infer_laughter(wav_path, checkpoint_path, config_path):
    """
    Loads a FusionModel checkpoint and predicts the laughter probability for a single WAV file.

    Args:
        wav_path (str): Path to the input WAV audio file.
        checkpoint_path (str): Path to the trained model checkpoint (.pt or .ckpt).
        config_path (str): Path to the YAML config file used during training (needed for model params).

    Returns:
        float: Probability of the audio containing laughter, or None if an error occurs.
    """
    if not os.path.exists(wav_path):
        print(f"Error: Audio file not found at {wav_path}")
        return None
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None

    config = load_yaml(config_path)
    if config is None:
        return None

    try:
        # --- Model Initialization ---
        model_params = config['model']
        num_classes_emotion = model_params.get('num_classes_emotion', 6) # Default if not in config
        num_classes_humor = model_params.get('num_classes_humor', 2) # Default if not in config
        hubert_model_name = model_params.get('hubert_model_name', "facebook/hubert-base-ls960")
        text_model_name = model_params.get('text_model_name', "distilbert-base-uncased") # Needed for init
        fusion_dim = model_params.get('fusion_dim', 512)
        dropout = model_params.get('dropout', 0.5)
        # Determine use_video/use_text flags based on config (or assume defaults if not critical for loading audio branch)
        use_video = model_params.get('use_video', False) # Assume False if not specified
        use_text = model_params.get('use_text', True) # Assume True if not specified


        # Instantiate the full model structure (even if only using audio part)
        # Pass dummy paths for video/hubert checkpoints if not strictly needed for loading structure
        # The actual weights will be loaded next.
        print("Initializing FusionModel structure...")
        model = FusionModel(
            num_classes_humor=num_classes_humor,
            num_classes_emotion=num_classes_emotion,
            video_checkpoint_path=None, # Not needed for structure if use_video=False
            hubert_checkpoint_path=None, # Weights loaded below
            hubert_model_name=hubert_model_name,
            text_model_name=text_model_name,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_video=use_video,
            use_text=use_text
        )
        print("FusionModel structure initialized.")

        # --- Load Checkpoint Weights ---
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Handle both raw state dict and dicts with 'model_state_dict' or 'state_dict' keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
             state_dict = checkpoint['state_dict']
             # Adjust keys if needed (e.g., remove 'model.' prefix if saved by PL)
             state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint # Assume it's just the state dict

        # Load weights, ignoring mismatches (e.g., if only audio branch weights are present)
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint state_dict. Load result (strict=False): {load_result}")
        if load_result.missing_keys or load_result.unexpected_keys:
            print("  Note: Mismatched keys are expected if checkpoint only contains partial model weights.")
            # print(f"  Missing keys: {load_result.missing_keys}") # Can be very verbose
            # print(f"  Unexpected keys: {load_result.unexpected_keys}")


        model.eval()  # Set to evaluation mode
        model.freeze() # Freeze weights

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded onto {device}.")

        # --- Load and Preprocess Audio ---
        target_sample_rate = config['data']['dataset_params'].get('sample_rate', 16000)
        print(f"Loading audio: {wav_path} (Target SR: {target_sample_rate} Hz)")
        waveform, sample_rate = torchaudio.load(wav_path)

        # Resample if necessary
        if sample_rate != target_sample_rate:
            print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Ensure mono channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Use Hubert feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(hubert_model_name)
        print(f"Preprocessing audio with {hubert_model_name} feature extractor...")
        # Process audio - ensure correct format and padding/truncation if needed
        # The feature extractor handles padding/truncation based on the model's requirements
        processed = feature_extractor(
            waveform.squeeze(0).numpy(), # Squeeze to 1D numpy array
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding=True # Pad or truncate to model's expected input size
        )
        audio_input_values = processed.input_values.to(device)
        audio_attention_mask = processed.attention_mask.to(device)
        print(f"Audio processed. Input shape: {audio_input_values.shape}")


        # --- Perform Inference ---
        print("Running inference...")
        with torch.no_grad():
            # Pass only necessary inputs for the laughter head (audio)
            # Model's forward pass handles None for unused branches
            model_outputs = model(audio_input_values=audio_input_values,
                                  audio_attention_mask=audio_attention_mask,
                                  video_input=None,
                                  text_input_ids=None,
                                  text_attention_mask=None)

            laugh_logits = model_outputs.get("laugh_logits")

            if laugh_logits is None:
                 print("Error: 'laugh_logits' not found in model output.")
                 return None

            prob = torch.sigmoid(laugh_logits.squeeze()) # Convert logit to probability

        laughter_probability = prob.item()
        print(f"Predicted laughter probability: {laughter_probability:.4f}")
        return laughter_probability

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laughter Inference Script")
    parser.add_argument('--wav', type=str, required=True, help='Path to the input WAV audio file.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pt or .ckpt) file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file used for training.')

    args = parser.parse_args()

    probability = infer_laughter(args.wav, args.ckpt, args.config)

    if probability is not None:
        print(f"\nFinal Result -> Laughter Probability for {os.path.basename(args.wav)}: {probability:.4f}")
    else:
        print("\nInference failed.")
