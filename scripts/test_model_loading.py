#!/usr/bin/env python3
"""
Test script for loading the emotion recognition model directly
"""

import os
import sys
import tensorflow as tf
import argparse
import h5py
import json
from tensorflow.keras.models import model_from_json

print("Using TensorFlow version:", tf.__version__)
print("Using Python version:", sys.version)

def direct_model_load(model_path):
    """
    Load the model using a direct file-based approach
    that avoids custom object scopes and compatibility issues
    """
    try:
        print(f"Loading model from {model_path}...")
        
        # Open the H5 file and extract the model architecture
        with h5py.File(model_path, 'r') as h5file:
            # Get model config
            if 'model_config' in h5file.attrs:
                model_config = h5file.attrs['model_config']
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                config_dict = json.loads(model_config)
                
                # Clean up the model config (remove time_major parameter from LSTM)
                def clean_config(conf):
                    if isinstance(conf, dict):
                        # Check for LSTM layer config
                        if conf.get('class_name') == 'LSTM' and 'config' in conf:
                            if 'time_major' in conf['config']:
                                del conf['config']['time_major']
                                print("Removed 'time_major' parameter from LSTM config")
                        
                        # Process all items in dict recursively
                        for k, v in conf.items():
                            if isinstance(v, dict):
                                clean_config(v)
                            elif isinstance(v, list):
                                for i in v:
                                    if isinstance(i, dict):
                                        clean_config(i)
                    return conf
                
                # Clean the configuration to remove incompatible parameters
                fixed_config = clean_config(config_dict)
                
                # Create the model from the cleaned config
                model = model_from_json(json.dumps(fixed_config))
                
                # Load the weights directly from the h5 file
                model.load_weights(model_path)
                
                # Compile the model (only needed for inference)
                model.compile(loss='categorical_crossentropy', optimizer='adam')
                
                print("Model loaded successfully with custom loading approach!")
                print(f"Input shapes: {[input.shape for input in model.inputs]}")
                print(f"Output shape: {model.output.shape}")
                
                return model
            else:
                raise ValueError("No model_config found in the H5 file")
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Test Model Loading')
    parser.add_argument('--model', type=str, default='models/branched_no_leakage_84_1/best_model.h5',
                       help='Path to the trained model')
    args = parser.parse_args()
    
    # Test if the model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        found_models = [f for f in os.listdir('models') if f.endswith('.h5')]
        if found_models:
            print("Available models:")
            for model in found_models:
                print(f"  - models/{model}")
        return

    print(f"Using model file: {args.model}")
    
    # Try to load the model
    model = direct_model_load(args.model)
    
    if model is not None:
        print("\nModel loading successful!")
        print("Model summary:")
        model.summary()
    else:
        print("\nModel loading failed.")

if __name__ == "__main__":
    main()
