#!/usr/bin/env python3
"""
Model-compatible version for loading the emotion recognition model
with improved TensorFlow version compatibility.
"""

import os
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_compatible_model(model_path):
    """
    Load a TensorFlow model with compatibility for different TF versions.
    Handles LSTM compatibility issues with 'time_major' and other parameters.
    
    Args:
        model_path: Path to the .h5 model file
    
    Returns:
        Loaded TensorFlow model or None if loading failed
    """
    # Try multiple approaches to load the model
    logging.info(f"Attempting to load model from {model_path}")
    
    # Try with custom objects specifically handling time_major
    try:
        # Define a robust LSTM compatibility layer to handle version differences
        class RobustLSTM(tf.keras.layers.LSTM):
            def __init__(self, *args, **kwargs):
                # Filter out incompatible kwargs
                compatible_kwargs = kwargs.copy()
                
                # Remove potentially incompatible parameters
                compatibility_params = [
                    'time_major', 
                    'dropout_state_filter_visitor',
                    'implementation'
                ]
                
                for param in compatibility_params:
                    if param in compatible_kwargs:
                        logging.info(f"Removing incompatible LSTM parameter: {param}={compatible_kwargs[param]}")
                        compatible_kwargs.pop(param)
                
                super(RobustLSTM, self).__init__(*args, **compatible_kwargs)
        
        # Try loading the model with custom objects
        custom_objects = {'LSTM': RobustLSTM}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        logging.info("Successfully loaded model with custom LSTM layer")
        return model
    
    except Exception as e:
        logging.warning(f"Failed to load with custom LSTM layer: {str(e)}")
    
    # Try a more advanced technique - modify the model file directly
    try:
        logging.info("Trying model loading with h5py modification technique")
        import h5py
        import tempfile
        import shutil
        
        # Create a temporary copy of the model file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_model_path = temp_file.name
        
        # Copy the original model to the temporary file
        shutil.copy2(model_path, temp_model_path)
        
        # Modify the h5 file to remove problematic attributes
        with h5py.File(temp_model_path, 'r+') as h5f:
            # Find LSTM layers by recursively searching groups
            def process_group(group):
                for key, item in group.items():
                    if isinstance(item, h5py.Group):
                        # If it's a group, search it recursively
                        process_group(item)
                    
                    # Look for LSTM layer config
                    if isinstance(item, h5py.Dataset) and 'lstm' in key.lower() and 'config' in key.lower():
                        try:
                            # Get the config data
                            config_data = item[()]
                            if isinstance(config_data, bytes):
                                import json
                                # Decode and parse the config
                                config_str = config_data.decode('utf-8')
                                config = json.loads(config_str)
                                
                                # Remove problematic parameters
                                if 'time_major' in config:
                                    logging.info(f"Removing time_major={config['time_major']} from LSTM config")
                                    del config['time_major']
                                
                                # Convert back to bytes and write to file
                                new_config_str = json.dumps(config)
                                item[()] = new_config_str.encode('utf-8')
                        except Exception as e:
                            logging.error(f"Error modifying LSTM config: {str(e)}")
            
            # Start processing from the root group
            process_group(h5f)
        
        # Try loading the modified model
        model = tf.keras.models.load_model(temp_model_path, compile=False)
        logging.info("Successfully loaded model with h5py modification")
        
        # Clean up the temporary file
        try:
            os.remove(temp_model_path)
        except:
            pass
            
        return model
        
    except Exception as e:
        logging.warning(f"Failed h5py modification approach: {str(e)}")
    
    # Last resort: Try with a TensorFlow version-specific approach
    try:
        tf_version = tf.__version__
        logging.info(f"Trying TensorFlow {tf_version} specific approach")
        
        # For TF 2.x, convert the model to JSON and load it differently
        temp_weights_path = 'temp_model_weights.h5'
        
        # Create a model with the same architecture but different LSTM implementation
        orig_model = tf.keras.models.load_model(model_path, compile=False, 
                                                 custom_objects={'LSTM': tf.compat.v1.keras.layers.LSTM})
        
        # Save just the weights
        orig_model.save_weights(temp_weights_path)
        
        # Get the model JSON and remove problematic parameters
        model_json = orig_model.to_json()
        model_json = model_json.replace('"time_major": false', '"time_major": null')
        model_json = model_json.replace('"time_major":false', '"time_major":null')
        
        # Create a new model from the modified JSON
        new_model = tf.keras.models.model_from_json(model_json)
        new_model.load_weights(temp_weights_path)
        
        # Clean up
        try:
            os.remove(temp_weights_path)
        except:
            pass
            
        logging.info("Successfully loaded model with JSON modification")
        return new_model
        
    except Exception as e:
        logging.error(f"All model loading approaches failed: {str(e)}")
        return None

def predict_emotion(model, face_features, audio_features):
    """
    Make an emotion prediction with the given model and features.
    
    Args:
        model: Loaded TensorFlow model
        face_features: Facial feature vector (512-d from FaceNet)
        audio_features: Audio feature vector (89-d from OpenSMILE)
        
    Returns:
        Numpy array of emotion probabilities
    """
    if model is None:
        # Return uniform distribution if no model
        return np.ones(6) / 6
    
    try:
        # Prepare the inputs (add batch dimension)
        face_batch = np.expand_dims(face_features, axis=0)
        audio_batch = np.expand_dims(audio_features, axis=0)
        
        # Check if the model expects time dimension
        # We can determine this by checking input shapes
        input_shapes = [input.shape for input in model.inputs]
        needs_time_dim = any(len(shape) > 2 for shape in input_shapes)
        
        if needs_time_dim:
            # Add time dimension (batch, time, features)
            face_batch = np.expand_dims(face_batch, axis=1)
            audio_batch = np.expand_dims(audio_batch, axis=1)
            logging.info("Added time dimension to features for prediction")
        
        # Make prediction based on number of inputs
        if len(model.inputs) == 2:
            # Model with separate audio and video inputs
            prediction = model.predict([face_batch, audio_batch], verbose=0)
        else:
            # Model with combined input
            combined = np.concatenate([face_batch, audio_batch], axis=-1)
            prediction = model.predict(combined, verbose=0)
        
        # Extract probabilities
        if isinstance(prediction, list):
            probs = prediction[0][0]
        else:
            probs = prediction[0]
        
        return probs
    
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return np.ones(6) / 6  # Return uniform distribution on error

if __name__ == "__main__":
    # Simple test for model loading
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/dynamic_padding_no_leakage/model_best.h5"
    
    print(f"Testing model compatibility with: {model_path}")
    model = load_compatible_model(model_path)
    
    if model is not None:
        print("\nModel loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model inputs: {[input.shape for input in model.inputs]}")
        print(f"Model outputs: {[output.shape for output in model.outputs]}")
        
        # Test with dummy data
        face_features = np.random.randn(512)
        audio_features = np.random.randn(89)
        
        probs = predict_emotion(model, face_features, audio_features)
        print("\nPrediction with random features:")
        for i, emotion in enumerate(["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]):
            print(f"{emotion}: {probs[i]:.4f}")
    else:
        print("Failed to load model")
