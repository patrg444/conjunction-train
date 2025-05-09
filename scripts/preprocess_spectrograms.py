#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesses RAVDESS and CREMA-D video files to extract audio,
compute Mel-spectrograms, and save them as NumPy arrays.
"""

import os
import sys # Import sys for sys.exit
import glob
import numpy as np
import librosa
import subprocess
import tempfile
import multiprocessing
import time
from tqdm import tqdm
import traceback # Import traceback for detailed error logging

# Configuration Variables (These will be set in __main__)
RAVDESS_VIDEO_DIR = ""
CREMA_D_VIDEO_DIR = ""
RAVDESS_OUTPUT_DIR = ""
CREMA_D_OUTPUT_DIR = ""

# Spectrogram Parameters
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 30 # Minimum frequency
FMAX = SAMPLE_RATE / 2 # Maximum frequency (Nyquist)

# Check for moviepy (optional, used as first attempt for audio extraction)
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
    print("MoviePy found, will use as primary audio extraction method.")
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("MoviePy not found. Will rely on ffmpeg (ensure it's installed and in PATH).")
except Exception as e:
    MOVIEPY_AVAILABLE = False
    print(f"Error importing MoviePy: {e}. Will rely on ffmpeg.")
    print(traceback.format_exc()) # Print full traceback

def extract_audio_moviepy(video_path, temp_wav_path):
    """Extracts audio using MoviePy."""
    try:
        print(f"  Attempting audio extraction with MoviePy for: {os.path.basename(video_path)}")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(temp_wav_path, codec='pcm_s16le', fps=SAMPLE_RATE, logger=None)
        video.close() # Close the video file handle
        print(f"    MoviePy extraction successful for: {os.path.basename(video_path)}")
        return True
    except Exception as e:
        print(f"  [MoviePy Error] Failed processing {os.path.basename(video_path)}: {e}")
        print(traceback.format_exc()) # Print full traceback
        # Clean up potentially incomplete temp file
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass
        return False

def extract_audio_ffmpeg(video_path, temp_wav_path):
    """Extracts audio using ffmpeg subprocess."""
    try:
        print(f"  Attempting audio extraction with FFmpeg for: {os.path.basename(video_path)}")
        command = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE), '-ac', '1', '-y', # -y overwrites output
            '-loglevel', 'error', # Suppress verbose output
            temp_wav_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True) # Added text=True
        print(f"    FFmpeg extraction successful for: {os.path.basename(video_path)}")
        return True
    except FileNotFoundError:
        print("  [FFmpeg Error] ffmpeg command not found. Ensure ffmpeg is installed and in your system PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  [FFmpeg Error] Failed processing {os.path.basename(video_path)}: {e}")
        print(f"  FFmpeg stderr: {e.stderr}") # No need for decode() with text=True
        print(traceback.format_exc()) # Print full traceback
        # Clean up potentially incomplete temp file
        if os.path.exists(temp_wav_path):
             try:
                 os.remove(temp_wav_path)
             except OSError:
                 pass
        return False
    except Exception as e:
        print(f"  [FFmpeg Error] Unexpected error processing {os.path.basename(video_path)}: {e}")
        print(traceback.format_exc()) # Print full traceback
        if os.path.exists(temp_wav_path):
             try:
                 os.remove(temp_wav_path)
             except OSError:
                 pass
        return False


def process_file(video_path_tuple):
    """Processes a single video file to extract Mel-spectrogram."""
    # Unpack tuple containing video path and global config paths
    video_path, ravdess_vid_dir_abs, cremad_vid_dir_abs, ravdess_out_dir_abs, cremad_out_dir_abs = video_path_tuple
    try:
        filename = os.path.basename(video_path)
        base_name, _ = os.path.splitext(filename)
        output_npy_path = ""
        print(f"\nProcessing file: {filename}")
        print(f"  Full video path: {video_path}")

        # Determine output path based on dataset
        if video_path.startswith(ravdess_vid_dir_abs): # Check if it's a RAVDESS file using absolute path
            # Get relative path from the *input* RAVDESS dir to find actor folder
            relative_path_from_input = os.path.relpath(video_path, ravdess_vid_dir_abs)
            actor_folder = os.path.dirname(relative_path_from_input) # Get actor folder relative path (e.g., "Actor_01")
            # Construct absolute output directory path
            output_dir = os.path.join(ravdess_out_dir_abs, actor_folder)
            output_npy_path = os.path.join(output_dir, base_name + ".npy")
            print(f"  Detected as RAVDESS. Actor folder: {actor_folder}")
        elif video_path.startswith(cremad_vid_dir_abs): # Check if it's a CREMA-D file using absolute path
            output_dir = cremad_out_dir_abs
            output_npy_path = os.path.join(output_dir, base_name + ".npy")
            print(f"  Detected as CREMA-D.")
        else:
            print(f"  Skipping unknown file structure: {video_path}")
            return f"Skipped (Unknown structure): {filename}"

        print(f"  Output directory: {output_dir}")
        print(f"  Output NPY path: {output_npy_path}")

        # Skip if already processed
        if os.path.exists(output_npy_path):
            print(f"  Skipping (Already Exists): {filename}")
            return f"Skipped (Exists): {filename}"

        # Create output directory if needed
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"  Ensured output directory exists: {output_dir}")
        except OSError as e:
            print(f"  [ERROR] Could not create output directory {output_dir}: {e}")
            print(traceback.format_exc())
            return f"Failed (Mkdir Error): {filename}"


        # Create temporary file for WAV
        temp_wav_path = "" # Initialize
        try:
            # Use mkstemp for better control and explicit deletion
            fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd) # Close the file descriptor opened by mkstemp
            print(f"  Created temporary WAV file: {temp_wav_path}")
        except Exception as e:
             print(f"  [ERROR] Could not create temporary WAV file: {e}")
             print(traceback.format_exc())
             return f"Failed (Temp File Error): {filename}"


        # --- Extract Audio ---
        audio_extracted = False
        if MOVIEPY_AVAILABLE:
            audio_extracted = extract_audio_moviepy(video_path, temp_wav_path)

        if not audio_extracted:
            # Try ffmpeg if moviepy failed or wasn't available
            audio_extracted = extract_audio_ffmpeg(video_path, temp_wav_path)

        if not audio_extracted or not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            print(f"  [ERROR] Failed to extract audio for {filename}")
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
            return f"Failed (Audio Extraction): {filename}"
        # --- End Audio Extraction ---

        # Load audio with librosa
        try:
            print(f"  Loading audio with Librosa from: {temp_wav_path}")
            y, sr = librosa.load(temp_wav_path, sr=SAMPLE_RATE, mono=True)
            print(f"    Librosa load successful. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
        except Exception as e:
            print(f"  [Librosa Load Error] Failed loading {temp_wav_path} for {filename}: {e}")
            print(traceback.format_exc())
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
            return f"Failed (Librosa Load): {filename}"
        finally:
             # Ensure temp file is deleted even if librosa fails
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                    print(f"  Deleted temporary WAV file: {temp_wav_path}")
                except OSError as e:
                     print(f"  Warning: Could not delete temp file {temp_wav_path}: {e}")


        # Compute Mel-spectrogram
        print("  Computing Mel-spectrogram...")
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX
        )
        print(f"    Mel-spectrogram shape: {mel_spec.shape}")

        # Convert to decibels (log scale)
        print("  Converting to dB scale...")
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Save as .npy
        print(f"  Saving to: {output_npy_path}")
        np.save(output_npy_path, mel_spec_db.astype(np.float32))
        print(f"    Successfully saved.")

        return f"Processed: {filename}"

    except Exception as e:
        print(f"  [ERROR] Unexpected error processing {video_path}: {e}")
        print(traceback.format_exc())
        # Attempt cleanup if temp file path was assigned
        if 'temp_wav_path' in locals() and temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass
        return f"Failed (Unexpected): {filename}"


if __name__ == '__main__':
    # Determine script's directory and project root assuming standard structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes scripts/ is one level down
    print(f"Detected Project Root: {project_root}")
    print(f"Script Directory: {script_dir}")

    # Set global paths relative to the project root
    RAVDESS_VIDEO_DIR = os.path.join(project_root, "data", "RAVDESS")
    CREMA_D_VIDEO_DIR = os.path.join(project_root, "data", "CREMA-D")
    RAVDESS_OUTPUT_DIR = os.path.join(project_root, "data", "ravdess_features_spectrogram")
    CREMA_D_OUTPUT_DIR = os.path.join(project_root, "data", "crema_d_features_spectrogram")

    # Get absolute paths for robust checking and processing
    ravdess_vid_dir_abs = os.path.abspath(RAVDESS_VIDEO_DIR)
    cremad_vid_dir_abs = os.path.abspath(CREMA_D_VIDEO_DIR)
    ravdess_out_dir_abs = os.path.abspath(RAVDESS_OUTPUT_DIR)
    cremad_out_dir_abs = os.path.abspath(CREMA_D_OUTPUT_DIR)

    print("\nStarting Mel-spectrogram preprocessing...")
    print(f"RAVDESS Source: {ravdess_vid_dir_abs}")
    print(f"CREMA-D Source: {cremad_vid_dir_abs}")
    print(f"RAVDESS Output: {ravdess_out_dir_abs}")
    print(f"CREMA-D Output: {cremad_out_dir_abs}")
    print(f"Parameters: SR={SAMPLE_RATE}, N_FFT={N_FFT}, HOP={HOP_LENGTH}, N_MELS={N_MELS}")

    # Check if source directories exist
    if not os.path.isdir(ravdess_vid_dir_abs):
        print(f"\nError: RAVDESS source directory not found: {ravdess_vid_dir_abs}")
        sys.exit(1)
    if not os.path.isdir(cremad_vid_dir_abs):
        print(f"\nError: CREMA-D source directory not found: {cremad_vid_dir_abs}")
        sys.exit(1)

    # Find video files using absolute paths derived from project root
    try:
        ravdess_pattern = os.path.join(ravdess_vid_dir_abs, "Actor_*", "*.mp4")
        crema_d_pattern = os.path.join(cremad_vid_dir_abs, "*.flv")
        print(f"Searching for RAVDESS files with pattern: {ravdess_pattern}")
        ravdess_files = glob.glob(ravdess_pattern)
        print(f"Searching for CREMA-D files with pattern: {crema_d_pattern}")
        crema_d_files = glob.glob(crema_d_pattern)
        all_files = ravdess_files + crema_d_files
    except Exception as e:
        print(f"\nError during glob file search: {e}")
        print(traceback.format_exc())
        sys.exit(1)


    if not all_files:
        print("\nError: No video files found in specified directories using glob patterns. Please check paths and permissions.")
        sys.exit(1) # Use sys.exit

    print(f"\nFound {len(ravdess_files)} RAVDESS files and {len(crema_d_files)} CREMA-D files.")
    print(f"Total files to process: {len(all_files)}")

    start_time = time.time()

    # Use multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() // 2) # Use half the CPU cores
    print(f"\nUsing {num_workers} worker processes...")

    # Prepare arguments for mapping: tuple of (video_path, config_paths...)
    process_args = [
        (f, ravdess_vid_dir_abs, cremad_vid_dir_abs, ravdess_out_dir_abs, cremad_out_dir_abs)
        for f in all_files
    ]

    results = []
    # Use try-except around the pool to catch potential pool errors
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=len(all_files), desc="Processing Videos") as pbar:
                for result in pool.imap_unordered(process_file, process_args): # Pass the tuple list
                    results.append(result)
                    pbar.update(1)
    except Exception as e:
        print(f"\n[ERROR] Multiprocessing pool encountered an error: {e}")
        print(traceback.format_exc())
        # Attempt to gather any results collected so far
        pass # results list will contain what was processed before the error

    end_time = time.time()
    print(f"\nPreprocessing finished in {end_time - start_time:.2f} seconds.")

    # Summary
    processed_count = sum(1 for r in results if r.startswith("Processed"))
    skipped_exist_count = sum(1 for r in results if r.startswith("Skipped (Exists)"))
    skipped_other_count = sum(1 for r in results if r.startswith("Skipped (Unknown structure)"))
    failed_count = sum(1 for r in results if r.startswith("Failed"))

    print("\nSummary:")
    print(f"- Successfully processed: {processed_count}")
    print(f"- Skipped (already exist): {skipped_exist_count}")
    print(f"- Skipped (unknown structure): {skipped_other_count}")
    print(f"- Failed: {failed_count}")

    if failed_count > 0:
        print("\nFiles that failed processing:")
        # Limit the number of failed files printed to avoid excessive output
        fail_limit = 20
        printed_fails = 0
        for r in results:
            if r.startswith("Failed"):
                print(f"- {r}")
                printed_fails += 1
                if printed_fails >= fail_limit:
                    print(f"... (omitting remaining {failed_count - printed_fails} failures)")
                    break
        # Indicate if the script should exit with an error code
        print("\nPreprocessing completed with errors.")
        sys.exit(1) # Exit with error code if failures occurred
    elif processed_count == 0 and skipped_exist_count == 0:
         print("\nWarning: No new files were processed and no existing files were found. Check input paths and file types.")
         sys.exit(1) # Exit with error code as likely something is wrong
    else:
        print("\nPreprocessing completed successfully.")
        sys.exit(0) # Explicitly exit with success code
