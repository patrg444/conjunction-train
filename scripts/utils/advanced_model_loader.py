#!/usr/bin/env python3
"""
Advanced Model Loader for TensorFlow
Handles compatibility issues between different TensorFlow versions
by directly modifying the model file if needed.
"""

import os
import json
import tempfile
import shutil
import h5py
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelLoader:
    """
    Advanced model loader that handles compatibility issues between
    different TensorFlow versions.
    """
    
    def __init__(self, debug=False):
        """Initialize the model loader."""
        self.debug = debug
        self.model = None
        self.model_path = None
        
        # Print debug info
        if debug:
            logging.info(f"TensorFlow version: {tf.__version__}")
            logging.info(f"Keras version: {tf.keras.__version__}")
            self._print_tf_config()
    
    def _print_tf_config(self):
        """Print TensorFlow configuration information."""
        # Available devices
        devices = tf.config.list_physical_devices()
        logging.info(f"Available TF devices: {[d.name for d in devices]}")
        
        # Memory growth status for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                memory_growth = tf.config.get_memory_growth(gpu)
                logging.info(f"Memory growth for {gpu.name}: {memory_growth}")
            except:
                logging.info(f"Could not get memory growth for {gpu.name}")
    
    def _fix_h5_file(self, src_path, dst_path):
        """
        Fix problematic parameters in the H5 model file.
        
        Args:
            src_path: Path to the source H5 file
            dst_path: Path to save the fixed H5 file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Copy file first
            shutil.copy2(src_path, dst_path)
            
            # Open the file for editing
            with h5py.File(dst_path, 'r+') as f:
                # Find all LSTM layer configs
                self._fix_group_configs(f)
            
            if self.debug:
                logging.info(f"Successfully fixed H5 file: {os.path.basename(dst_path)}")
            return True
            
        except Exception as e:
            logging.error(f"Error fixing H5 file: {str(e)}")
            return False
    
    def _fix_group_configs(self, group, path=""):
        """
        Recursively fix all LSTM configs in a group.
        
        Args:
            group: The h5py group to fix
            path: Current path in the hierarchy (for debugging)
        """
        for key in group.keys():
            item = group[key]
            current_path = f"{path}/{key}"
            
            # If it's a dataset containing config data
            if isinstance(item, h5py.Dataset) and ('config' in key and 
                                                 ('lstm' in current_path.lower() or 
                                                  'rnn' in current_path.lower())):
                try:
                    # Try to load and fix config
                    config_data = item[()]
                    if isinstance(config_data, bytes):
                        # Decode the config
                        config_str = config_data.decode('utf-8')
                        config = json.loads(config_str)
                        
                        # List of parameters to remove
                        params_to_remove = [
                            'time_major',
                            'dropout_state_filter_visitor',
                            'implementation'
                        ]
                        
                        # Track what we've modified
                        modified = False
                        for param in params_to_remove:
                            if param in config:
                                if self.debug:
                                    logging.info(f"Removing '{param}' with value {config[param]} from {current_path}")
                                del config[param]
                                modified = True
                        
                        # Write back the modified config
                        if modified:
                            new_config_str = json.dumps(config)
                            item[()] = new_config_str.encode('utf-8')
                            if self.debug:
                                logging.info(f"Updated config at {current_path}")
                except Exception as e:
                    logging.error(f"Error processing config at {current_path}: {str(e)}")
            
            # If it's a group, recursively process it
            elif isinstance(item, h5py.Group):
                self._fix_group_configs(item, current_path)
    
    def load_model(self, model_path, force_fix=False):
        """
        Load a model with compatibility fixes.
        
        Args:
            model_path: Path to the model file
            force_fix: Whether to force fixing the model file even if not necessary
        
        Returns:
            The loaded model or None if loading failed
        """
        self.model_path = model_path
        
        # First try direct loading
        if not force_fix:
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
                logging.info(f"Successfully loaded model from {model_path} without fixes")
                return self.model
            except Exception as e:
                logging.warning(f"Direct loading failed: {str(e)}")
        
        # If direct loading failed or force_fix is True, try fixing the model file
        logging.info("Attempting to fix model file...")
        
        # Create a temporary file for the fixed model
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_model_path = temp_file.name
        
        # Fix the H5 file
        success = self._fix_h5_file(model_path, temp_model_path)
        if not success:
            logging.error("Failed to fix model file")
            return None
        
        # Try loading the fixed model
        try:
            self.model = tf.keras.models.load_model(temp_model_path, compile=False)
            logging.info(f"Successfully loaded fixed model from {temp_model_path}")
            return self.model
        except Exception as e:
            logging.error(f"Failed to load fixed model: {str(e)}")
            
            # Try a more aggressive approach - custom objects
            try:
                # Define a custom LSTM layer that ignores problematic parameters
                class CustomLSTM(tf.keras.layers.LSTM):
                    def __init__(self, *args, **kwargs):
                        # Remove problematic kwargs
                        for param in ['time_major', 'dropout_state_filter_visitor', 'implementation']:
                            kwargs.pop(param, None) 
                        super(CustomLSTM, self).__init__(*args, **kwargs)
                
                # Try loading with custom objects
                custom_objects = {'LSTM': CustomLSTM}
                self.model = tf.keras.models.load_model(
                    temp_model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
                logging.info("Successfully loaded model with custom LSTM layer")
                return self.model
            except Exception as e:
                logging.error(f"Failed to load with custom objects: {str(e)}")
        
        # Clean up temp file
        try:
            os.remove(temp_model_path)
        except:
            pass
        
        return None
    
    def get_input_shapes(self):
        """Get the input shapes for the model."""
        if self.model is None:
            return None
        
        return [input.shape for input in self.model.inputs]
    
    def get_output_shapes(self):
        """Get the output shapes for the model."""
        if self.model is None:
            return None
        
        return [output.shape for output in self.model.outputs]
    
    def predict(self, inputs, verbose=0):
        """
        Make a prediction with the model.
        
        Args:
            inputs: Input data (single input or list for multiple inputs)
            verbose: Verbosity level
        
        Returns:
            Model predictions or None if prediction failed
        """
        if self.model is None:
            logging.error("No model loaded")
            return None
        
        try:
            return self.model.predict(inputs, verbose=verbose)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return None

# Direct usage example
if __name__ == "__main__":
    import sys
    
    # Get model path from command line
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/dynamic_padding_no_leakage/model_best.h5"
    
    # Initialize loader
    loader = ModelLoader(debug=True)
    
    # Load model
    model = loader.load_model(model_path)
    
    if model is not None:
        print("\nModel loaded successfully!")
        print(f"Model input shapes: {loader.get_input_shapes()}")
        print(f"Model output shapes: {loader.get_output_shapes()}")
        
        # Create dummy inputs for testing
        input_shapes = loader.get_input_shapes()
        dummy_inputs = []
        
        for shape in input_shapes:
            # Create a batch of random data with the right shape
            # Assuming first dimension is batch size
            dummy_shape = list(shape)
            dummy_shape[0] = 1  # Batch size of 1
            
            # Handle time dimension if present
            if len(dummy_shape) > 2:
                dummy_inputs.append(np.random.random(dummy_shape))
            else:
                # If no time dimension but model expects it, add it
                dummy_inputs.append(np.random.random(dummy_shape))
        
        # Make a test prediction
        try:
            if len(dummy_inputs) > 1:
                preds = model.predict(dummy_inputs, verbose=0)
            else:
                preds = model.predict(dummy_inputs[0], verbose=0)
            
            print("\nTest prediction succeeded!")
            if isinstance(preds, list):
                for i, pred in enumerate(preds):
                    print(f"Output {i} shape: {pred.shape}")
            else:
                print(f"Output shape: {preds.shape}")
        except Exception as e:
            print(f"\nTest prediction failed: {str(e)}")
    else:
        print("\nFailed to load model")
