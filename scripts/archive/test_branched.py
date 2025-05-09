#!/usr/bin/env python3
"""
Test script for running the branched LSTM model with a smaller number of epochs
to verify it works with openSMILE features.
"""

import os
import sys
import logging
import argparse
from train_branched import main as train_branched_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_branched.log"),
        logging.StreamHandler()
    ]
)

def override_train_branched_params():
    """Override the train_branched parameters to use fewer epochs.
    
    This works by modifying the train_multimodal_model function to use
    our test parameters instead of the defaults defined in main().
    """
    import train_branched
    
    # Store the original training function
    original_train_fn = train_branched.train_multimodal_model
    
    # Create a wrapper function that overrides parameters
    def train_wrapper(*args, **kwargs):
        # Override epochs and batch_size with our test values
        kwargs['epochs'] = 5  # Use only 5 epochs for testing
        kwargs['batch_size'] = 2  # Use smaller batch size
        
        # Call the original function with modified kwargs
        return original_train_fn(*args, **kwargs)
    
    # Replace the training function
    train_branched.train_multimodal_model = train_wrapper

def main():
    parser = argparse.ArgumentParser(description='Test the branched LSTM model with openSMILE features')
    parser.add_argument('--data-dir', type=str, default='processed_features',
                       help='Directory containing processed features')
    parser.add_argument('--output-dir', type=str, default='model_evaluation/branched_test',
                       help='Directory to save evaluation results')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs to train for testing (default: 5)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for training (default: 2)')
    
    args = parser.parse_args()
    
    # Print test configuration
    print(f"\n=== Testing Branched LSTM Model with OpenSMILE Features ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}\n")
    
    # Override training parameters
    override_train_branched_params()
    
    try:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run the modified train_branched main
        train_branched_main()
        
        print("\n✅ Test completed successfully! The branched model works with openSMILE features.")
        print(f"Evaluation results saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
