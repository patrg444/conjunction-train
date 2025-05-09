#!/usr/bin/env python
"""
Model Loader Fix for TensorFlow Compatibility

This script demonstrates how to load LSTM-based models that use the time_major parameter
in a way that's compatible with newer TensorFlow versions where this parameter 
has been changed or deprecated.
"""

import tensorflow as tf
import os
import logging
import argparse
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibleLSTM(tf.keras.layers.LSTM):
    """Custom LSTM layer that ignores the time_major parameter for compatibility."""
    
    def __init__(self, *args, **kwargs):
        # Remove time_major if present (will be ignored)
        if 'time_major' in kwargs:
            logger.info(f"Removing 'time_major' parameter with value {kwargs['time_major']}")
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

class ModelLoader:
    """Handles loading TensorFlow models with special handling for LSTM compatibility issues."""
    
    @staticmethod
    def load_with_custom_objects(model_path):
        """Load model with custom objects to handle compatibility issues."""
        try:
            # First attempt: Try direct loading
            logger.info(f"Attempt 1: Direct model loading from {model_path}")
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully with direct loading!")
            return model, "direct"
        except Exception as e:
            logger.warning(f"Direct model loading failed: {str(e)}")
            
            try:
                # Second attempt: Use custom objects with compatible LSTM
                logger.info("Attempt 2: Loading with custom CompatibleLSTM")
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'LSTM': CompatibleLSTM}
                )
                logger.info("Model loaded successfully with CompatibleLSTM!")
                return model, "compatible_lstm"
            except Exception as e:
                logger.warning(f"CompatibleLSTM loading failed: {str(e)}")
                
                try:
                    # Third attempt: Try TF compatibility mode
                    logger.info("Attempt 3: Loading with TensorFlow compatibility mode")
                    with tf.keras.utils.custom_object_scope({'LSTM': CompatibleLSTM}):
                        model = tf.keras.models.load_model(model_path)
                    logger.info("Model loaded successfully with TF compatibility mode!")
                    return model, "tf_compat"
                except Exception as e:
                    logger.error(f"All loading attempts failed: {str(e)}")
                    return None, "failed"

def main():
    parser = argparse.ArgumentParser(description='Load and test emotion recognition model with compatibility fixes')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (.h5)')
    args = parser.parse_args()
    
    # Print system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")

    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    # Try to load the model
    model, method = ModelLoader.load_with_custom_objects(args.model)
    
    if model is None:
        logger.error("Failed to load model after all attempts")
        return 1
    
    # Output model summary and loading method
    logger.info(f"Model loaded successfully using method: {method}")
    logger.info("Model Summary:")
    model.summary(print_fn=logger.info)
    
    # Test model input shape
    input_shapes = []
    for layer in model.inputs:
        input_shapes.append(layer.shape)
    logger.info(f"Model input shapes: {input_shapes}")
    
    # Test model output shape
    output_shapes = []
    for layer in model.outputs:
        output_shapes.append(layer.shape)
    logger.info(f"Model output shapes: {output_shapes}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
