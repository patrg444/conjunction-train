#!/usr/bin/env python3
"""
Run the enhanced emotion recognition pipeline from data processing to model training.
This script provides a convenient way to execute the complete pipeline with the
improved anti-overfitting measures.
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def run_command(command, description=None):
    """Run a shell command and log the output.
    
    Args:
        command: Shell command to run
        description: Optional description of the command
        
    Returns:
        True if command succeeded, False otherwise
    """
    if description:
        logging.info(f"=== {description} ===")
    
    logging.info(f"Running command: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print and log output in real-time
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            logging.info(line)
            
        process.wait()
        
        if process.returncode == 0:
            logging.info(f"Command completed successfully (exit code 0)")
            return True
        else:
            logging.error(f"Command failed with exit code {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Error running command: {str(e)}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run enhanced emotion recognition pipeline')
    
    parser.add_argument('--dataset', type=str, default='data/RAVDESS',
                        help='Directory containing RAVDESS dataset')
    
    parser.add_argument('--features-dir', type=str, default='processed_features_3_5s',
                        help='Directory to save processed features')
    
    parser.add_argument('--min-segments', type=int, default=1000,
                        help='Minimum number of segments to generate')
    
    parser.add_argument('--window-size', type=float, default=3.5,
                        help='Time window size in seconds (default: 3.5 seconds)')
    
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers for processing (None for auto-detect)')
    
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip dataset processing step (use existing features)')
    
    parser.add_argument('--models', type=str, default='both', choices=['both', 'branched', 'dual_stream'],
                        help='Which model(s) to train: both, branched, or dual_stream (default: both)')
    
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable attention mechanisms in branched model to reduce complexity')
    
    return parser.parse_args()

def main():
    """Run the complete enhanced pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create needed directories
    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs("models/branched_enhanced", exist_ok=True)
    os.makedirs("models/dual_stream_enhanced", exist_ok=True)
    os.makedirs("model_evaluation/branched_enhanced", exist_ok=True) 
    os.makedirs("model_evaluation/dual_stream_enhanced", exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"ENHANCED EMOTION RECOGNITION PIPELINE")
    print("=" * 70)
    print(f"Dataset directory: {args.dataset}")
    print(f"Features directory: {args.features_dir}")
    print(f"Window size: {args.window_size} seconds")
    print(f"Model training: {args.models.upper()}")
    print(f"Skip processing: {args.skip_processing}")
    print("=" * 70 + "\n")
    
    # Track overall success
    success = True
    
    # Step 1: Process RAVDESS dataset (if not skipped)
    if not args.skip_processing:
        workers_arg = f"--workers {args.workers}" if args.workers is not None else ""
        
        process_command = (
            f"python scripts/process_ravdess_dataset.py "
            f"--dataset {args.dataset} "
            f"--output {args.features_dir} "
            f"--min-segments {args.min_segments} "
            f"--window-size {args.window_size} "
            f"{workers_arg} "
            f"--visualize"
        )
        
        if not run_command(process_command, "PROCESSING RAVDESS DATASET"):
            print("\n❌ Dataset processing failed. Check logs for errors.")
            return False
    else:
        logging.info("Skipping dataset processing as requested")
        
        # Verify that the features directory exists
        if not os.path.exists(args.features_dir):
            logging.error(f"Features directory {args.features_dir} not found. Cannot skip processing.")
            print(f"\n❌ Features directory {args.features_dir} not found. Cannot skip processing.")
            return False
            
        # Check if there are any .npz files
        import glob
        npz_files = glob.glob(os.path.join(args.features_dir, "*.npz"))
        if not npz_files:
            logging.error(f"No .npz files found in {args.features_dir}. Cannot skip processing.")
            print(f"\n❌ No .npz files found in {args.features_dir}. Cannot skip processing.")
            return False
            
        logging.info(f"Found {len(npz_files)} existing feature files in {args.features_dir}")
    
    # Step 2: Train enhanced branched model
    if args.models in ['both', 'branched']:
        attention_arg = "--no-attention" if args.no_attention else ""
        
        branched_command = (
            f"python scripts/train_branched_enhanced.py "
            f"--data_dir {args.features_dir} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--model_dir models/branched_enhanced/ "
            f"--eval_dir model_evaluation/branched_enhanced/ "
            f"{attention_arg}"
        )
        
        if not run_command(branched_command, "TRAINING ENHANCED BRANCHED MODEL"):
            print("\n❌ Branched model training failed. Check logs for errors.")
            success = False
    
    # Step 3: Train enhanced dual-stream model
    if args.models in ['both', 'dual_stream']:
        dual_stream_command = (
            f"python scripts/train_dual_stream_enhanced.py "
            f"--data_dir {args.features_dir} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--model_dir models/dual_stream_enhanced/ "
            f"--eval_dir model_evaluation/dual_stream_enhanced/"
        )
        
        if not run_command(dual_stream_command, "TRAINING ENHANCED DUAL-STREAM MODEL"):
            print("\n❌ Dual-stream model training failed. Check logs for errors.")
            success = False
    
    # Final status report
    print("\n" + "=" * 70)
    if success:
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        
        # Print locations of model files and evaluation results
        print("\nGenerated files:")
        print(f"- Processed features: {args.features_dir}/")
        
        if args.models in ['both', 'branched']:
            print(f"- Branched model: models/branched_enhanced/final_branched_model_enhanced.h5")
            print(f"- Branched model evaluation: model_evaluation/branched_enhanced/")
            
        if args.models in ['both', 'dual_stream']:
            print(f"- Dual-stream model: models/dual_stream_enhanced/final_model_enhanced.h5")
            print(f"- Dual-stream model evaluation: model_evaluation/dual_stream_enhanced/")
    else:
        print("❌ PIPELINE COMPLETED WITH ERRORS")
        print("   Check the logs for details on what went wrong.")
        
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    main()
