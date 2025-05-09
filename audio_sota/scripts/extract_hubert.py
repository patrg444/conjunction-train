import os
import argparse
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf
import torch.nn.functional as F

def load_audio(file_path, target_sr=16000):
    """Loads audio file, resamples if necessary, and ensures mono."""
    try:
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1: # Convert to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0) # Remove channel dim for processor
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

def extract_features(
    dataset_name, 
    split_csv, 
    data_root_dir, # Directory relative to which paths in CSV are defined (e.g., audio_sota/data)
    output_feat_dir, 
    checkpoint="facebook/hubert-large-ls960-ft", 
    batch_size=8, 
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Extracts HuBERT features for audio files listed in a CSV split."""
    
    csv_path = Path(split_csv)
    data_root = Path(data_root_dir)
    output_dir = Path(output_feat_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: Split CSV file not found at {csv_path}")
        return
        
    print(f"Loading model {checkpoint}...")
    model = HubertModel.from_pretrained(checkpoint).to(device)
    model.eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
    target_sr = processor.sampling_rate
    
    print(f"Using device: {device}")
    print(f"Target sample rate: {target_sr}")
    
    df = pd.read_csv(csv_path)
    file_paths = df['path'].tolist()
    
    print(f"Found {len(file_paths)} files in {csv_path}. Starting feature extraction...")
    
    num_batches = (len(file_paths) + batch_size - 1) // batch_size
    
    # Removed with torch.no_grad() wrapper from the loop
    for i in tqdm(range(num_batches), desc=f"Extracting features ({csv_path.stem})"):
        # --- Start of indented block --- Corrected Indentation ---
        batch_paths_relative = file_paths[i*batch_size : (i+1)*batch_size]
        
        # --- MODIFIED: Robust path construction --- Final Corrected Logic ---
        batch_paths_absolute = []
        for p_str in batch_paths_relative:
            p = Path(p_str)
            # Check if the relative path already starts with the dataset name directory
            # Needed because RAVDESS CSVs include 'ravdess/', CREMA-D CSVs don't include 'crema_d/'
            if p.parts and p.parts[0] == dataset_name:
                # If it does (like in RAVDESS CSVs), join directly with data_root
                abs_path = data_root / p
            else:
                # If it doesn't (like in CREMA-D CSVs), join data_root, dataset_name, and the relative path
                abs_path = data_root / dataset_name / p
            batch_paths_absolute.append(abs_path)
        # --- END MODIFICATION ---
        
        waveforms = [load_audio(p, target_sr=target_sr) for p in batch_paths_absolute]
        valid_indices = [idx for idx, wf in enumerate(waveforms) if wf is not None]
        
        if not valid_indices:
            print(f"Warning: Skipping batch {i+1} due to audio loading errors.")
            continue # Skip to next iteration of the outer loop

        valid_waveforms = [waveforms[idx] for idx in valid_indices]
        valid_paths_relative = [batch_paths_relative[idx] for idx in valid_indices]

        # --- RE-ADDED Manual Padding ---
        if not valid_waveforms: # Should not happen due to earlier check, but safety first
             print(f"Warning: No valid waveforms in batch {i+1} after filtering Nones.")
             continue # Skip to next iteration of the outer loop

        max_len = max(wf.shape[0] for wf in valid_waveforms)
        padded_waveforms = []
        for wf in valid_waveforms:
            padding_needed = max_len - wf.shape[0]
            if padding_needed > 0:
                # Pad on the right (dim=0 for 1D tensor)
                padded_wf = F.pad(wf, (0, padding_needed), "constant", 0)
            else:
                padded_wf = wf
            padded_waveforms.append(padded_wf)
        # --- End Manual Padding ---

        # --- RE-ADDED Stacking ---
        # Stack the list of 1D tensors into a 2D tensor [batch_size, max_len]
        input_tensor = torch.stack(padded_waveforms) # Shape: [batch_size, max_len]
        # --- End Stacking ---

        # --- MODIFIED: Manually create attention mask ---
        attention_mask = torch.zeros_like(input_tensor, dtype=torch.long)
        for idx, wf in enumerate(valid_waveforms):
             # Set 1s for the actual length of the original waveform
             attention_mask[idx, :wf.shape[0]] = 1 
        # --- End Attention Mask Creation ---

        # --- MODIFIED: Bypass processor for main input, construct input dict manually ---
        inputs = {
            "input_values": input_tensor.to(device),
            "attention_mask": attention_mask.to(device)
        }
        # --- END MODIFICATION ---

        # Extract features (last hidden state) - Apply no_grad here
        with torch.no_grad():
             outputs = model(**inputs)
        features = outputs.last_hidden_state # Shape: (batch_size, sequence_length, hidden_size)

        # Save features for each valid file in the batch
        for j, feat_tensor in enumerate(features):
            original_relative_path = valid_paths_relative[j]
            # Construct output path mirroring the input structure relative to data_root
            # e.g., crema_d/AudioWAV/Actor_01/file.wav -> hubert_large/crema_d/AudioWAV/Actor_01/file.pt
            # e.g., ravdess/AudioWAV/Actor_01/file.wav -> hubert_large/ravdess/AudioWAV/Actor_01/file.pt
            output_subpath = Path(original_relative_path).with_suffix('.pt')
            output_file_path = output_dir / output_subpath
            
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Detach, move to CPU (if on GPU), and save
            torch.save(feat_tensor.detach().cpu(), output_file_path)
        # --- End of indented block ---

    print(f"Feature extraction complete for {csv_path.stem}. Features saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HuBERT features for audio datasets.")
    parser.add_argument('--dataset', required=True, choices=['crema_d', 'ravdess'], help='Name of the dataset.')
    parser.add_argument('--split_csv', required=True, help='Path to the train/validation/test CSV file.')
    parser.add_argument('--data_root', required=True, help='Root directory where paths in CSV are relative to (e.g., audio_sota/data).')
    parser.add_argument('--output_dir', required=True, help='Directory to save the extracted features.')
    parser.add_argument('--checkpoint', default='facebook/hubert-large-ls960-ft', help='Hugging Face model checkpoint.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    
    args = parser.parse_args()
    
    extract_features(
        args.dataset, 
        args.split_csv, 
        args.data_root,
        args.output_dir, 
        checkpoint=args.checkpoint, 
        batch_size=args.batch_size, 
        device=args.device
    )
