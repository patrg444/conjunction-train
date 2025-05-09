#!/usr/bin/env python3
"""
Preprocess audio spectrograms to extract CNN features
Fixed version that handles shape issues correctly
"""

import os
import sys
import argparse
import glob
import time
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import needed components
try:
    from scripts.spectrogram_cnn_pooling_generator import build_cnn_feature_extractor
    from scripts.preprocess_spectrograms import N_MELS
except ImportError as e:
    print(f"Error importing required components: {e}")
    sys.exit(1)

# Setup logging
def setup_logging(log_dir="logs", verbose=False):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocess_cnn_audio_{timestamp}.log")
    
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_file(file_path, full_output_path, cnn_model):
    """
    Process a single spectrogram file to extract CNN features

    Arguments:
        file_path: Path to the input spectrogram npy file
        full_output_path: The complete path (including filename) where the output feature should be saved
        cnn_model: The loaded CNN feature extractor model

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the spectrogram
        spectrogram = np.load(file_path)
        
        # Print original shape for debugging
        print(f"Original spectrogram shape: {spectrogram.shape}")
        
        # FIXED SHAPE HANDLING:
        # The CNN model expects shape (batch, time, frequency, channel) = (None, None, 128, 1)
        # Our data is in shape (batch=1, frequency=128, time=variable)
        
        # Case 1: If data is already in 3D format (batch, freq, time)
        if len(spectrogram.shape) == 3:
            # Reshape to (batch, time, freq, channel)
            spectrogram = np.transpose(spectrogram, (0, 2, 1))  # Now (batch, time, freq)
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension -> (batch, time, freq, channel)
        
        # Case 2: If data is in 2D format (freq, time) or (time, freq)
        elif len(spectrogram.shape) == 2:
            # Check if it's (freq, time) or (time, freq)
            if spectrogram.shape[0] == N_MELS:  # If first dim is frequency (N_MELS=128)
                # It's in (freq, time) format, need to transpose
                spectrogram = spectrogram.T  # Now (time, freq)
            
            # Add batch and channel dimensions
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch -> (batch, time, freq)
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel -> (batch, time, freq, channel)
        
        # Print reshaped dimensions for debugging
        print(f"Reshaped spectrogram: {spectrogram.shape}")
        
        # Extract CNN features
        features = cnn_model.predict(spectrogram, verbose=0)
        
        # Ensure output directory exists just in case
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

        # Save the features to the full specified path
        np.save(full_output_path, features[0]) # Save the actual features (index 0 removes batch dim)

        return True
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False

def process_files_worker(args):
    """Worker function for multiprocessing"""
    file_path, full_output_path, model_path = args # Unpack the full output path

    # Load CNN model inside worker process
    cnn_input_shape = (None, N_MELS, 1) # Assuming N_MELS is globally accessible or defined
    # It might be more efficient to pass the model object if using spawn context,
    # but for fork (default on Linux), loading per worker is safer.
    cnn_model = build_cnn_feature_extractor(input_shape=cnn_input_shape)

    # Process the file using the full output path
    return process_file(file_path, full_output_path, cnn_model)

def main():
    parser = argparse.ArgumentParser(description="Extract CNN features from spectrograms")
    parser.add_argument("--spectrogram_dir", type=str, default=None, 
                        help="Directory containing spectrogram files")
    parser.add_argument("--crema_d_dir", type=str, default="data/crema_d_features_spectrogram",
                        help="Directory for CREMA-D spectrogram files")
    parser.add_argument("--ravdess_dir", type=str, default="data/ravdess_features_spectrogram",
                        help="Directory for RAVDESS spectrogram files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for CNN features")
    parser.add_argument("--crema_d_output", type=str, default="data/crema_d_features_cnn_audio",
                        help="Output directory for CREMA-D CNN features")
    parser.add_argument("--ravdess_output", type=str, default="data/ravdess_features_cnn_audio",
                        help="Output directory for RAVDESS CNN features")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation of features even if they exist")
    parser.add_argument("--files", nargs='+', default=None,
                        help="List of specific spectrogram file paths to process (overrides directory scan)")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    logger.info("Starting CNN feature extraction")
    
    # Force CPU-only mode to avoid GPU memory issues with multiprocessing
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    logger.info("Running in CPU-only mode")

    # Determine input and output directories
    input_dir_to_process = None
    output_dir_to_process = None

    if args.files:
        # If specific files are given, output_dir must also be given
        if not args.output_dir:
             logger.error("Output directory (--output_dir) must be specified when using --files")
             return
        output_dir_to_process = args.output_dir
        # Input directory isn't strictly needed here, but we set it for consistency if possible
        if args.spectrogram_dir:
             input_dir_to_process = args.spectrogram_dir
        else:
             # Try to infer from the first file path if possible
             if args.files:
                 input_dir_to_process = os.path.dirname(args.files[0])

    elif args.spectrogram_dir and args.output_dir:
        # Use the explicitly provided directories
        input_dir_to_process = args.spectrogram_dir
        output_dir_to_process = args.output_dir
    else:
        # Fallback or error if no directories/files are specified correctly
        # This case should ideally not be reached if called from the shell script correctly
        logger.error("Please specify --spectrogram_dir and --output_dir, or use --files with --output_dir.")
        return

    # Ensure the single output directory exists
    if output_dir_to_process:
        os.makedirs(output_dir_to_process, exist_ok=True)
    else:
        # This case should also not be reached if arguments are correct
        logger.error("Could not determine output directory.")
        return

    # Get all files to process
    all_file_tasks = []
    if args.files:
        # Process specific files provided via --files argument
        if not args.output_dir:
            logger.error("Output directory (--output_dir) must be specified when using --files")
            return
        logger.info(f"Processing {len(args.files)} specific files provided via --files argument.")
        spectrogram_files = args.files
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

        if not args.force:
             # Skip files that already have output
            remaining_files = []
            for file_path in spectrogram_files:
                file_name = os.path.basename(file_path)
                output_path = os.path.join(output_dir, file_name)
                if not os.path.exists(output_path):
                    remaining_files.append(file_path)
            
            logger.info(f"Skipping {len(spectrogram_files) - len(remaining_files)} files that already have output")
            spectrogram_files = remaining_files

        # Add tasks for the specified files
        for file_path in spectrogram_files:
             # Ensure the input file exists before adding task
             if os.path.exists(file_path):
                 all_file_tasks.append((file_path, output_dir, None))
             else:
                 logger.warning(f"Specified file not found, skipping: {file_path}")

    else:
        # Process the determined input directory if --files was not used
        if input_dir_to_process and not args.files:
            if not os.path.exists(input_dir_to_process):
                logger.warning(f"Input directory does not exist: {input_dir_to_process}")
                return # Exit if the main directory doesn't exist

            # Find all spectrogram files recursively
            search_pattern = os.path.join(input_dir_to_process, "**", "*.npy")
            spectrogram_files = glob.glob(search_pattern, recursive=True)
            logger.info(f"Found {len(spectrogram_files)} files recursively in {input_dir_to_process}")

            if not args.force:
                 # Skip files that already have output
                 remaining_files = []
                 for file_path in spectrogram_files:
                     # Construct output path relative to the base output directory
                     relative_path = os.path.relpath(file_path, input_dir_to_process)
                     output_path = os.path.join(output_dir_to_process, relative_path)
                     # Ensure the subdirectory exists in the output path
                     os.makedirs(os.path.dirname(output_path), exist_ok=True)
                     if not os.path.exists(output_path):
                         remaining_files.append(file_path)

                 logger.info(f"Skipping {len(spectrogram_files) - len(remaining_files)} files that already have output")
                 spectrogram_files = remaining_files

            # Add tasks for this directory
            for file_path in spectrogram_files:
                 # Construct output path relative to the base output directory
                 relative_path = os.path.relpath(file_path, input_dir_to_process)
                 output_path_for_task = os.path.join(output_dir_to_process, relative_path)
                 # The actual saving happens in process_file, which now needs the full path
                 all_file_tasks.append((file_path, output_path_for_task, None)) # Pass the full path

    if not all_file_tasks:
        logger.info("No files to process")
        return
    
    logger.info(f"Processing {len(all_file_tasks)} files with {args.workers} workers")
    
    # Process files using multiprocessing
    with mp.Pool(processes=args.workers) as pool:
        # Create a shared CNN model for workers
        cnn_input_shape = (None, N_MELS, 1)
        # No need to load model here if loading per-worker
        # cnn_input_shape = (None, N_MELS, 1)
        # cnn_model = build_cnn_feature_extractor(input_shape=cnn_input_shape)

        # Worker arguments now contain the full output path
        worker_args = all_file_tasks # Already in the correct format (file_path, full_output_path, None)

        # Process files with progress bar
        results = []
        for i, result in enumerate(tqdm(
            pool.imap(process_files_worker, worker_args),
            total=len(worker_args),
            desc="Extracting CNN Features"
        )):
            results.append(result)
    
    # Count successes
    success_count = sum(1 for r in results if r)
    logger.info(f"Processed {success_count} files successfully out of {len(all_file_tasks)}")
    logger.info("Feature extraction completed")

if __name__ == "__main__":
    main()
