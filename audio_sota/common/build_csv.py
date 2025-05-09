import os
import pandas as pd
import argparse
import re
from pathlib import Path
from tqdm import tqdm

def parse_crema_d_filename(filename):
    """Parses CREMA-D filename to extract metadata."""
    # Example: 1001_DFA_ANG_XX.wav
    # ActorID_SentenceID_Emotion_Intensity.wav
    parts = filename.stem.split('_')
    if len(parts) != 4:
        return None
    actor_id, sentence_id, emotion_code, intensity_code = parts
    
    emotion_map = {
        'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 
        'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'
    }
    intensity_map = {'LO': 'low', 'MD': 'medium', 'HI': 'high', 'XX': 'unspecified'}
    
    emotion = emotion_map.get(emotion_code)
    intensity = intensity_map.get(intensity_code)
    
    if not emotion or not intensity:
        return None
        
    return {
        'speaker': actor_id,
        'emotion': emotion,
        'intensity': intensity,
        'sentence': sentence_id,
        'split': 'unknown' # Will be filled later
    }

def parse_ravdess_filename(filepath):
    """Parses RAVDESS filename and path to extract metadata."""
    # Example filepath: audio_sota/data/ravdess/AudioWAV/Actor_01/03-01-01-01-01-01-01.wav
    # We need actor from path and codes from filename stem.
    filename = Path(filepath)
    parts = filename.stem.split('-')
    
    # Extract Actor ID from the parent directory name (e.g., Actor_01 -> 01)
    actor_dir_name = filename.parent.name
    actor_match = re.match(r'Actor_(\d+)', actor_dir_name)
    if not actor_match:
        # print(f"Debug: Could not extract actor ID from path: {filepath}")
        return None
    actor_id = actor_match.group(1) # This should be the padded actor ID like '01', '24'

    if len(parts) != 7:
        # print(f"Debug: Filename '{filename.name}' does not have 7 parts.")
        return None
        
    modality, vocal_channel, emotion_code, intensity_code, statement, repetition, _ = parts # Ignore actor ID from filename part
    
    # We are processing WAV files extracted from audio modality (03) and speech channel (01)
    # Let's keep the check but make it less strict or log if it fails unexpectedly
    if modality != '03':
        # print(f"Warning: Unexpected modality '{modality}' in filename {filename.name}")
        pass # Allow processing anyway, assuming it's audio
    if vocal_channel != '01':
        # print(f"Warning: Unexpected vocal channel '{vocal_channel}' in filename {filename.name}")
        pass # Allow processing anyway

    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
        '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
    }
    intensity_map = {'01': 'normal', '02': 'strong'}
    statement_map = {'01': 'kids', '02': 'dogs'}

    emotion = emotion_map.get(emotion_code)
    intensity = intensity_map.get(intensity_code)
    statement_type = statement_map.get(statement)

    if not emotion or not intensity or not statement_type:
        # print(f"Debug: Failed map lookup for {filename.name}. Codes: E={emotion_code}, I={intensity_code}, S={statement}")
        return None

    return {
        'speaker': actor_id, # Extracted from path
        'emotion': emotion,
        'intensity': intensity,
        'sentence': statement_type, # 'kids' or 'dogs'
        'repetition': repetition, # '01' or '02'
        'split': 'unknown'
    }


def build_metadata(dataset_name, raw_data_dir, output_dir):
    """Scans raw data directory (containing WAVs), parses filenames/paths, and saves metadata CSV."""
    raw_path = Path(raw_data_dir) # Should point to the directory containing WAV files (e.g., AudioWAV)
    output_path = Path(output_dir) # Should point to the dataset root (e.g., audio_sota/data/ravdess)
    output_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_path / 'metadata.csv'

    metadata_list = []

    print(f"Scanning {raw_path} for {dataset_name} audio files...")

    # Use appropriate parser based on dataset name
    if dataset_name == 'crema_d':
        # TODO: Update to use VideoDemographics.csv per SOTA doc if file becomes available. Currently uses filename parsing.
        parser_func = parse_crema_d_filename
        wav_files = list(raw_path.rglob('*.wav')) # Search recursively for .wav files (Aligned with SOTA doc workflow)
    elif dataset_name == 'ravdess':
        parser_func = parse_ravdess_filename
        # Search recursively within the AudioWAV dir (which contains Actor_* subdirs)
        wav_files = list(raw_path.rglob('*.wav')) # Search recursively for .wav files (Aligned with SOTA doc workflow)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not wav_files:
         print(f"Error: No .wav files found in {raw_path}") # Updated error message
         return

    print(f"Found {len(wav_files)} potential audio files. Parsing metadata...")

    files_parsed = 0
    files_failed = 0
    for wav_file_path in tqdm(wav_files, desc=f"Parsing {dataset_name}"):
        # Pass the full path to the parser function
        metadata = parser_func(wav_file_path)
        if metadata:
            # Store path relative to the *output* directory's parent (i.e., relative to audio_sota/data/)
            try:
                 relative_path = str(wav_file_path.relative_to(output_path.parent))
                 metadata['path'] = relative_path
                 metadata_list.append(metadata)
                 files_parsed += 1
            except ValueError:
                 print(f"Warning: Could not make path relative: {wav_file_path} relative to {output_path.parent}")
                 files_failed += 1
        else:
            # Parser function should ideally handle logging warnings/errors for specific files
            # print(f"Warning: Could not parse metadata for: {wav_file_path.name}")
            files_failed += 1

    print(f"Parsing complete. Parsed: {files_parsed}, Failed: {files_failed}")

    if not metadata_list:
        print("Error: No valid metadata could be extracted after parsing.")
        return

    df = pd.DataFrame(metadata_list)

    # Define expected columns for ordering
    crema_cols = ['path', 'speaker', 'emotion', 'intensity', 'sentence', 'split']
    ravdess_cols = ['path', 'speaker', 'emotion', 'intensity', 'sentence', 'repetition', 'split']

    # Ensure consistent column order and handle potential missing columns gracefully
    if dataset_name == 'crema_d':
        cols = crema_cols
    else: # ravdess
        cols = ravdess_cols
    
    # Reindex DataFrame to ensure all expected columns are present, filling missing ones with NaN if necessary
    df = df.reindex(columns=cols)

    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")
    print(f"Total valid samples processed: {files_parsed}")
    if files_parsed > 0:
        print("\nEmotion distribution:")
        print(df['emotion'].value_counts())
        print("\nSpeaker distribution:")
        print(df['speaker'].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build metadata CSV from extracted WAV files.")
    parser.add_argument('--dataset', required=True, choices=['crema_d', 'ravdess'], help='Name of the dataset.')
    parser.add_argument('--raw_dir', required=True, help='Path to the directory containing extracted WAV files (e.g., .../AudioWAV).')
    parser.add_argument('--output_dir', required=True, help='Path to the dataset root directory where metadata.csv will be saved (e.g., audio_sota/data/crema_d).')

    args = parser.parse_args()

    build_metadata(args.dataset, args.raw_dir, args.output_dir)
