#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesses Mel-spectrograms using the CNN feature extractor defined in
spectrogram_cnn_pooling_generator.py and saves the extracted features.
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # Added import
import multiprocessing
import time
from tqdm import tqdm
import traceback

# Import the CNN builder function and constants from the generator script
try:
    # Temporarily add script directory to path to ensure import works
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    # Import CNN builder and constants from the correct scripts
    from spectrogram_cnn_pooling_generator import build_cnn_feature_extractor, AUDIO_CNN_FILTERS
    from preprocess_spectrograms import N_MELS # Import N_MELS from its source
    print("Successfully imported CNN builder and N_MELS.")
except ImportError as e:
    print(f"Error importing required components: {e}")
    print("Ensure spectrogram_cnn_pooling_generator.py is in the same directory or Python path.")
    sys.exit(1)
finally:
    # Clean up sys.path if modified
    if _script_dir in sys.path and _script_dir == sys.path[0]:
        sys.path.pop(0)

# --- Configuration ---
# Input directories (where spectrogram .npy files are)
RAVDESS_SPEC_INPUT_DIR = ""
CREMA_D_SPEC_INPUT_DIR = ""
# Output directories (where CNN audio features will be saved)
RAVDESS_CNN_OUTPUT_DIR = ""
CREMA_D_CNN_OUTPUT_DIR = ""

# --- CNN Feature Extractor ---
# Build the model once
try:
    CNN_INPUT_SHAPE = (None, N_MELS, 1) # Time dimension is variable
    print("Building CNN feature extractor model...")
    CNN_EXTRACTOR_MODEL = build_cnn_feature_extractor(CNN_INPUT_SHAPE)
    CNN_EXTRACTOR_MODEL.summary(line_length=100)
    CNN_OUTPUT_DIM = AUDIO_CNN_FILTERS[-1]
    print(f"CNN model built successfully. Output dimension: {CNN_OUTPUT_DIM}")
except Exception as e:
    print(f"Error building CNN model: {e}")
    print(traceback.format_exc())
    sys.exit(1)

def process_spectrogram_file(spec_path_tuple):
    """Loads a spectrogram, extracts CNN features, and saves them."""
    # Unpack tuple
    spec_path, ravdess_spec_in_abs, cremad_spec_in_abs, ravdess_cnn_out_abs, cremad_cnn_out_abs = spec_path_tuple
    try:
        filename = os.path.basename(spec_path)
        base_name, _ = os.path.splitext(filename)
        output_npy_path = ""
        # print(f"\nProcessing spectrogram: {filename}")

        # Determine output path based on dataset
        if spec_path.startswith(ravdess_spec_in_abs):
            relative_path_from_input = os.path.relpath(spec_path, ravdess_spec_in_abs)
            actor_folder = os.path.dirname(relative_path_from_input)
            output_dir = os.path.join(ravdess_cnn_out_abs, actor_folder)
            output_npy_path = os.path.join(output_dir, base_name + ".npy")
        elif spec_path.startswith(cremad_spec_in_abs):
            output_dir = cremad_cnn_out_abs
            output_npy_path = os.path.join(output_dir, base_name + ".npy")
        else:
            # print(f"  Skipping unknown structure: {spec_path}")
            return f"Skipped (Unknown structure): {filename}"

        # Skip if already processed
        if os.path.exists(output_npy_path):
            # print(f"  Skipping (Already Exists): {filename}")
            return f"Skipped (Exists): {filename}"

        # Create output directory if needed
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"  [ERROR] Could not create output directory {output_dir}: {e}")
            print(traceback.format_exc())
            return f"Failed (Mkdir Error): {filename}"

        # Load spectrogram
        try:
            # print(f"  Loading spectrogram: {spec_path}")
            # Spectrograms are saved as (n_mels, time_frames) in preprocess_spectrograms.py
            spec_db = np.load(spec_path).astype(np.float32)
            # Transpose to (time_frames, n_mels) for CNN input consistency
            spec_db = np.transpose(spec_db, (1, 0))
            # print(f"    Loaded shape: {spec_db.shape}")
        except Exception as e:
            print(f"  [ERROR] Failed loading {spec_path}: {e}")
            print(traceback.format_exc())
            return f"Failed (Load Error): {filename}"

        if spec_db.shape[0] == 0: # Check if spectrogram is empty
             print(f"  [Warning] Spectrogram is empty: {filename}")
             return f"Skipped (Empty Spectrogram): {filename}"

        # Prepare for CNN: Add channel and batch dimensions
        # Input shape expected: (batch, time_steps, n_mels, channels)
        cnn_input = np.expand_dims(spec_db, axis=-1) # (time_frames, n_mels, 1)
        cnn_input = np.expand_dims(cnn_input, axis=0) # (1, time_frames, n_mels, 1)
        # print(f"    CNN Input shape: {cnn_input.shape}")

        # Extract features using the CNN model
        try:
            # print("  Extracting features with CNN...")
            # Use predict for potentially larger inputs, though predict_on_batch might work too
            cnn_features = CNN_EXTRACTOR_MODEL.predict(tf.convert_to_tensor(cnn_input, dtype=tf.float32), verbose=0)
            # Output shape from GlobalAveragePooling2D is (batch, filters)
            # BUT we ran it on the whole sequence, so the output should reflect time if we didn't use GAP?
            # Let's re-check the generator's CNN model... it uses GlobalAveragePooling2D.
            # This means it outputs ONE vector per spectrogram slice.
            # If we feed the WHOLE spectrogram, it will average over the whole time axis.
            # This is NOT what we want. We need features PER TIME STEP (or pooled per video frame).

            # --- Correction: Need features per frame BEFORE GlobalAveragePooling ---
            # Let's get the output of the layer *before* GlobalAveragePooling2D
            intermediate_layer_model = Model(inputs=CNN_EXTRACTOR_MODEL.input,
                                             outputs=CNN_EXTRACTOR_MODEL.layers[-2].output) # Output of Dropout before GAP
            cnn_features_per_frame = intermediate_layer_model.predict(tf.convert_to_tensor(cnn_input, dtype=tf.float32), verbose=0)
            # Output shape: (1, time_steps_cnn, freq_bins_cnn, filters)
            # Squeeze the batch dimension
            cnn_features_per_frame = np.squeeze(cnn_features_per_frame, axis=0)
            # print(f"    CNN features per frame shape: {cnn_features_per_frame.shape}")

            # We need to map these CNN time steps back to the original spectrogram time steps.
            # The CNN downsamples time by a factor of 2^3 = 8 due to 3 MaxPooling layers.
            # Let's save these features directly. The generator will handle alignment.
            # Shape to save: (time_steps_cnn, freq_bins_cnn, filters)

        except Exception as e:
            print(f"  [ERROR] CNN prediction failed for {filename}: {e}")
            print(traceback.format_exc())
            return f"Failed (CNN Predict): {filename}"

        # Save features
        try:
            # print(f"  Saving CNN features to: {output_npy_path}")
            np.save(output_npy_path, cnn_features_per_frame.astype(np.float32))
            # print(f"    Successfully saved.")
        except Exception as e:
            print(f"  [ERROR] Failed saving features for {filename} to {output_npy_path}: {e}")
            print(traceback.format_exc())
            return f"Failed (Save Error): {filename}"

        return f"Processed: {filename}"

    except Exception as e:
        print(f"  [ERROR] Unexpected error processing {spec_path}: {e}")
        print(traceback.format_exc())
        return f"Failed (Unexpected): {filename}"


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    print(f"Detected Project Root: {project_root}")

    # Set global paths relative to the project root
    RAVDESS_SPEC_INPUT_DIR = os.path.join(project_root, "data", "ravdess_features_spectrogram")
    CREMA_D_SPEC_INPUT_DIR = os.path.join(project_root, "data", "crema_d_features_spectrogram")
    RAVDESS_CNN_OUTPUT_DIR = os.path.join(project_root, "data", "ravdess_features_cnn_audio") # New output dir
    CREMA_D_CNN_OUTPUT_DIR = os.path.join(project_root, "data", "crema_d_features_cnn_audio") # New output dir

    # Get absolute paths
    ravdess_spec_in_abs = os.path.abspath(RAVDESS_SPEC_INPUT_DIR)
    cremad_spec_in_abs = os.path.abspath(CREMA_D_SPEC_INPUT_DIR)
    ravdess_cnn_out_abs = os.path.abspath(RAVDESS_CNN_OUTPUT_DIR)
    cremad_cnn_out_abs = os.path.abspath(CREMA_D_CNN_OUTPUT_DIR)

    print("\nStarting CNN Audio Feature preprocessing...")
    print(f"RAVDESS Spectrogram Source: {ravdess_spec_in_abs}")
    print(f"CREMA-D Spectrogram Source: {cremad_spec_in_abs}")
    print(f"RAVDESS CNN Feature Output: {ravdess_cnn_out_abs}")
    print(f"CREMA-D CNN Feature Output: {cremad_cnn_out_abs}")

    # Check if source directories exist
    if not os.path.isdir(ravdess_spec_in_abs):
        print(f"\nError: RAVDESS spectrogram source directory not found: {ravdess_spec_in_abs}")
        sys.exit(1)
    if not os.path.isdir(cremad_spec_in_abs):
        print(f"\nError: CREMA-D spectrogram source directory not found: {cremad_spec_in_abs}")
        sys.exit(1)

    # Find spectrogram files
    try:
        ravdess_pattern = os.path.join(ravdess_spec_in_abs, "Actor_*", "*.npy")
        crema_d_pattern = os.path.join(cremad_spec_in_abs, "*.npy")
        ravdess_files = glob.glob(ravdess_pattern)
        crema_d_files = glob.glob(crema_d_pattern)
        all_spec_files = ravdess_files + crema_d_files
    except Exception as e:
        print(f"\nError during glob file search: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    if not all_spec_files:
        print("\nError: No spectrogram files found in specified directories.")
        sys.exit(1)

    print(f"\nFound {len(ravdess_files)} RAVDESS and {len(crema_d_files)} CREMA-D spectrogram files.")
    print(f"Total files to process: {len(all_spec_files)}")

    start_time = time.time()

    # Use multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"\nUsing {num_workers} worker processes...")

    # Prepare arguments for mapping
    process_args = [
        (f, ravdess_spec_in_abs, cremad_spec_in_abs, ravdess_cnn_out_abs, cremad_cnn_out_abs)
        for f in all_spec_files
    ]

    results = []
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=len(all_spec_files), desc="Extracting CNN Features") as pbar:
                for result in pool.imap_unordered(process_spectrogram_file, process_args):
                    results.append(result)
                    pbar.update(1)
    except Exception as e:
        print(f"\n[ERROR] Multiprocessing pool encountered an error: {e}")
        print(traceback.format_exc())

    end_time = time.time()
    print(f"\nPreprocessing finished in {end_time - start_time:.2f} seconds.")

    # Summary
    processed_count = sum(1 for r in results if r.startswith("Processed"))
    skipped_exist_count = sum(1 for r in results if r.startswith("Skipped (Exists)"))
    skipped_other_count = sum(1 for r in results if r.startswith("Skipped (Unknown structure)") or r.startswith("Skipped (Empty Spectrogram)"))
    failed_count = sum(1 for r in results if r.startswith("Failed"))

    print("\nSummary:")
    print(f"- Successfully processed: {processed_count}")
    print(f"- Skipped (already exist): {skipped_exist_count}")
    print(f"- Skipped (other): {skipped_other_count}")
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
        print("\nPreprocessing completed with errors.")
        sys.exit(1)
    elif processed_count == 0 and skipped_exist_count == 0:
         print("\nWarning: No new files were processed and no existing files were found. Check input paths.")
         sys.exit(1)
    else:
        print("\nPreprocessing completed successfully.")
        sys.exit(0)
