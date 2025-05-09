#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts Wav2Vec2 embeddings from raw audio in RAVDESS and CREMA-D datasets.
Assumes raw audio/video files are located in specified input directories.
Saves embeddings as NumPy arrays. Runs sequentially.
"""

import os
import sys
import glob
import numpy as np
import librosa
import subprocess
import tempfile
from joblib import Parallel, delayed # Re-enable joblib
import time
import argparse
import torch # Assuming PyTorch backend for transformers
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import traceback
import functools # Re-enable functools for partial

# --- Configuration (Adjustable via argparse) ---
TARGET_SAMPLE_RATE = 16000 # Wav2Vec2 models typically expect 16kHz
MODEL_NAME = "facebook/wav2vec2-base-960h" # Default Wav2Vec2 model

# --- Audio Extraction (Adapted from preprocess_spectrograms.py) ---
# Check for moviepy (optional, used as first attempt for audio extraction)
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
    # Keep print statement in main block
except ImportError:
    MOVIEPY_AVAILABLE = False
except Exception as e:
    MOVIEPY_AVAILABLE = False
    print(f"Warning: Error importing MoviePy: {e}. Will rely on ffmpeg.")
    # print(traceback.format_exc()) # Less verbose

def extract_audio_moviepy(video_path, temp_wav_path):
    """Extracts audio using MoviePy."""
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(temp_wav_path, codec='pcm_s16le', fps=TARGET_SAMPLE_RATE, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"  [MoviePy Error] Failed processing {os.path.basename(video_path)}: {e}")
        if os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except OSError: pass
        return False

def extract_audio_ffmpeg(video_path, temp_wav_path):
    """Extracts audio using ffmpeg subprocess."""
    try:
        command = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(TARGET_SAMPLE_RATE), '-ac', '1', '-y',
            '-loglevel', 'error', temp_wav_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        print("  [FFmpeg Error] ffmpeg command not found.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  [FFmpeg Error] Failed processing {os.path.basename(video_path)}: {e}")
        print(f"  FFmpeg stderr: {e.stderr}")
        if os.path.exists(temp_wav_path):
             try: os.remove(temp_wav_path)
             except OSError: pass
        return False
    except Exception as e:
        print(f"  [FFmpeg Error] Unexpected error processing {os.path.basename(video_path)}: {e}")
        if os.path.exists(temp_wav_path):
             try: os.remove(temp_wav_path)
             except OSError: pass
        return False

# --- Wav2Vec2 Feature Extraction ---
def process_audio_file(audio_path, processor, model, device): # Added processor, model, device args
    """Loads audio, processes it, and extracts Wav2Vec2 embeddings."""
    try:
        waveform, sample_rate = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        if sample_rate != TARGET_SAMPLE_RATE:
             waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

        input_values = processor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt").input_values
        input_values = input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            embeddings = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy() # Detach added

        return embeddings.astype(np.float32)
    except Exception as e:
        print(f"  [ERROR] Failed to process audio file {audio_path}: {e}")
        return None

# --- Main Processing Function ---
def process_video_file_to_wav2vec(video_path_tuple, processor, model, device): # Added processor, model, device args
    """
    Processes a single video file: extracts audio, then extracts Wav2Vec2 embeddings.
    """
    video_path, input_base_dir_abs, output_base_dir_abs = video_path_tuple
    temp_wav_path = ""
    try:
        filename = os.path.basename(video_path)
        base_name, _ = os.path.splitext(filename)

        relative_path = os.path.relpath(video_path, input_base_dir_abs)
        output_dir = os.path.join(output_base_dir_abs, os.path.dirname(relative_path))
        output_npy_path = os.path.join(output_dir, base_name + ".npy")

        if os.path.exists(output_npy_path):
            return f"Skipped (Exists): {filename}"

        os.makedirs(output_dir, exist_ok=True)
        fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        audio_extracted = False
        if MOVIEPY_AVAILABLE:
            audio_extracted = extract_audio_moviepy(video_path, temp_wav_path)
        if not audio_extracted:
            audio_extracted = extract_audio_ffmpeg(video_path, temp_wav_path)

        if not audio_extracted or not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            print(f"  [ERROR] Failed to extract audio for {filename}")
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
            return f"Failed (Audio Extraction): {filename}"

        embeddings = process_audio_file(temp_wav_path, processor, model, device) # Pass components

        if os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except OSError as e: print(f"  Warning: Could not delete temp file {temp_wav_path}: {e}")

        if embeddings is None:
            return f"Failed (Wav2Vec2 Extraction): {filename}"

        np.save(output_npy_path, embeddings)
        return f"Processed: {filename}"

    except Exception as e:
        print(f"  [ERROR] Unexpected error processing {video_path}: {e}")
        print(traceback.format_exc())
        if temp_wav_path and os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except OSError: pass
        return f"Failed (Unexpected): {filename}"
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except OSError: pass


# --- Main Execution ---
if __name__ == '__main__':
    # Removed multiprocessing setup

    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 embeddings from RAVDESS and CREMA-D audio.")
    parser.add_argument("--ravdess_dir", default="/home/ec2-user/data/RAVDESS", help="Path to the raw RAVDESS video files directory.")
    parser.add_argument("--cremad_dir", default="/home/ec2-user/data/CREMA-D", help="Path to the raw CREMA-D video files directory.")
    parser.add_argument("--ravdess_out_dir", default="/home/ec2-user/emotion_training/data/ravdess_features_wav2vec2", help="Output directory for RAVDESS Wav2Vec2 embeddings.")
    parser.add_argument("--cremad_out_dir", default="/home/ec2-user/emotion_training/data/crema_d_features_wav2vec2", help="Output directory for CREMA-D Wav2Vec2 embeddings.")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Hugging Face model name (default: {MODEL_NAME}).")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1 for sequential).")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to first N files per dataset (for testing).")

    args = parser.parse_args()

    MODEL_NAME = args.model # Update global model name if provided

    # Print MoviePy status
    if MOVIEPY_AVAILABLE:
        print("MoviePy is available.")
    else:
        print("MoviePy not found, relying on ffmpeg.")

    # Initialize the model ONCE in the main process
    print(f"Initializing Wav2Vec2 model: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model loaded on device: {device}")

    # Get absolute paths
    ravdess_dir_abs = os.path.abspath(args.ravdess_dir)
    cremad_dir_abs = os.path.abspath(args.cremad_dir)
    ravdess_out_dir_abs = os.path.abspath(args.ravdess_out_dir)
    cremad_out_dir_abs = os.path.abspath(args.cremad_out_dir)

    print("\nStarting Wav2Vec2 embedding extraction...")
    print(f"RAVDESS Source: {ravdess_dir_abs}")
    print(f"CREMA-D Source: {cremad_dir_abs}")
    print(f"RAVDESS Output: {ravdess_out_dir_abs}")
    print(f"CREMA-D Output: {cremad_out_dir_abs}")
    print(f"Model: {MODEL_NAME}")
    print(f"Target Sample Rate: {TARGET_SAMPLE_RATE}")

    # Check source directories
    if not os.path.isdir(ravdess_dir_abs):
        print(f"\nError: RAVDESS source directory not found: {ravdess_dir_abs}")
        sys.exit(1)
    if not os.path.isdir(cremad_dir_abs):
        print(f"\nError: CREMA-D source directory not found: {cremad_dir_abs}")
        sys.exit(1)

    # Find video files
    try:
        ravdess_pattern = os.path.join(ravdess_dir_abs, "Actor_*", "*.mp4")
        crema_d_pattern = os.path.join(cremad_dir_abs, "*.flv") # CREMA-D uses .flv
        print(f"Searching for RAVDESS files: {ravdess_pattern}")
        ravdess_files = glob.glob(ravdess_pattern)
        print(f"Searching for CREMA-D files: {crema_d_pattern}")
        crema_d_files = glob.glob(crema_d_pattern)
    except Exception as e:
        print(f"\nError during glob file search: {e}")
        sys.exit(1)

    if not ravdess_files and not crema_d_files:
        print("\nError: No video files found in specified directories.")
        sys.exit(1)

    print(f"\nFound {len(ravdess_files)} RAVDESS files and {len(crema_d_files)} CREMA-D files.")

    # Prepare arguments
    process_args = []
    if ravdess_files:
        ravdess_subset = ravdess_files[:args.limit] if args.limit else ravdess_files
        process_args.extend([(f, ravdess_dir_abs, ravdess_out_dir_abs) for f in ravdess_subset])
        print(f"Processing {len(ravdess_subset)} RAVDESS files.")
    if crema_d_files:
        cremad_subset = crema_d_files[:args.limit] if args.limit else crema_d_files
        process_args.extend([(f, cremad_dir_abs, cremad_out_dir_abs) for f in cremad_subset])
        print(f"Processing {len(cremad_subset)} CREMA-D files.")

    if not process_args:
        print("No files selected for processing (check limit?).")
        sys.exit(0)

    print(f"Total files to process: {len(process_args)}")
    start_time = time.time()

    # Determine number of workers
    num_workers = args.workers
    if num_workers <= 0:
        num_workers = 1 # Ensure at least 1 worker for sequential fallback
    elif num_workers > os.cpu_count():
        print(f"Warning: Requested workers ({num_workers}) exceeds CPU count ({os.cpu_count()}). Setting to {os.cpu_count()}.")
        num_workers = os.cpu_count()

    # Create a partial function with fixed model components
    # This avoids pickling large model objects for each worker
    process_partial = functools.partial(process_video_file_to_wav2vec,
                                        processor=processor,
                                        model=model,
                                        device=device)

    if num_workers > 1:
        print(f"\nProcessing files in parallel with {num_workers} workers...")
        # Use joblib for parallel processing
        results = Parallel(n_jobs=num_workers, backend="multiprocessing", verbose=10)(
            delayed(process_partial)(args_tuple) for args_tuple in process_args
        )
    else:
        # Process files sequentially in the main process if workers=1
        print(f"\nProcessing files sequentially...")
        results = []
        with tqdm(total=len(process_args), desc="Extracting Wav2Vec2 Embeddings") as pbar:
            for args_tuple in process_args:
                result = process_partial(args_tuple)
                results.append(result)
                pbar.update(1)

    end_time = time.time()
    print(f"\nExtraction finished in {end_time - start_time:.2f} seconds.")

    # Summary
    processed_count = sum(1 for r in results if r.startswith("Processed"))
    skipped_exist_count = sum(1 for r in results if r.startswith("Skipped (Exists)"))
    failed_count = sum(1 for r in results if r.startswith("Failed"))

    print("\nSummary:")
    print(f"- Successfully processed: {processed_count}")
    print(f"- Skipped (already exist): {skipped_exist_count}")
    print(f"- Failed: {failed_count}")

    if failed_count > 0:
        print("\nFiles that failed processing:")
        fail_limit = 20
        printed_fails = 0
        for r in results:
            if r.startswith("Failed"):
                print(f"- {r}")
                printed_fails += 1
                if printed_fails >= fail_limit:
                    print(f"... (omitting remaining {failed_count - printed_fails} failures)")
                    break
        print("\nExtraction completed with errors.")
        sys.exit(1)
    elif processed_count == 0 and skipped_exist_count == 0:
         print("\nWarning: No new files were processed and no existing files were found.")
         sys.exit(1)
    else:
        print("\nExtraction completed successfully.")
        sys.exit(0)
