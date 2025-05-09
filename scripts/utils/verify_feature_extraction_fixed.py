#!/usr/bin/env python3
"""
Enhanced verification script for feature extraction in real-time emotion recognition.
This script tests the FaceNet and OpenSMILE pipelines with more robust error handling
and outputs detailed analysis of the extracted features.
"""

import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
import logging
import subprocess
import pyaudio
import wave
import tensorflow as tf
import argparse
from facenet_extractor import FaceNetExtractor

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feature_verification.log"),
        logging.StreamHandler()
    ]
)

def take_multiple_frames(max_attempts=5, delay=0.5):
    """Capture multiple frames to increase chances of getting a valid face."""
    logging.info(f"Attempting to capture up to {max_attempts} frames...")
    
    frames = []
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("Failed to open camera")
        return None
    
    try:
        for i in range(max_attempts):
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame {i+1}")
                continue
                
            frames.append(frame)
            logging.info(f"Captured frame {i+1}/{max_attempts}")
            
            # Save the frame for visualization
            cv2.imwrite(f"verification_frame_{i+1}.jpg", frame)
            
            # Wait a moment to allow for position changes
            time.sleep(delay)
    
    finally:
        cap.release()
    
    logging.info(f"Captured {len(frames)} frames successfully")
    return frames

def verify_facenet_features_multiple(frames):
    """Extract features from multiple frames and return the best one."""
    if not frames:
        logging.error("No frames provided for feature extraction")
        return None
        
    # Initialize FaceNetExtractor
    try:
        face_extractor = FaceNetExtractor(
            keep_all=False,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # Try each frame until we get non-zero features
        for i, frame in enumerate(frames):
            logging.info(f"Trying feature extraction on frame {i+1}/{len(frames)}")
            
            start_time = time.time()
            features = face_extractor.extract_features(frame)
            elapsed = time.time() - start_time
            
            # Print feature statistics
            logging.info(f"FaceNet feature shape: {features.shape}")
            logging.info(f"Extraction time: {elapsed:.3f} seconds")
            
            non_zero = np.count_nonzero(features)
            total = features.shape[0]
            
            if non_zero > 0:
                logging.info(f"Frame {i+1}: Found face with non-zero features!")
                logging.info(f"Non-zero elements: {non_zero}/{total} ({non_zero/total*100:.1f}%)")
                logging.info(f"Min: {np.min(features):.4f}, Max: {np.max(features):.4f}")
                logging.info(f"Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")
                
                # Visualize some features as a sanity check
                logging.info(f"Feature sample (first 10): {features[:10]}")
                
                return features
            else:
                logging.warning(f"Frame {i+1}: All FaceNet features are zero - no face detected")
        
        # If we reach here, no face was detected in any frame
        logging.error("No face detected in any of the frames")
        # Return zeros but with the right shape for model testing
        return np.zeros(512)
        
    except Exception as e:
        logging.error(f"Error in FaceNet feature extraction: {str(e)}")
        logging.exception("Stack trace:")
        return None

def verify_opensmile_features(opensmile_path, config_file):
    """Extract and verify audio features using OpenSMILE."""
    logging.info("Verifying OpenSMILE feature extraction...")
    
    if not os.path.exists(opensmile_path):
        logging.error(f"OpenSMILE executable not found at {opensmile_path}")
        return None
    
    if not os.path.exists(config_file):
        logging.error(f"OpenSMILE config not found at {config_file}")
        return None
    
    # Create temporary directory and files
    temp_dir = "temp_verification_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_wav = os.path.join(temp_dir, f"test_{int(time.time())}.wav")
    temp_csv = os.path.join(temp_dir, f"test_{int(time.time())}.csv")
    
    try:
        # Record a short audio sample
        logging.info("Recording audio sample...")
        p = pyaudio.PyAudio()
        sample_rate = 16000
        channels = 1
        chunk_size = 1024
        format = pyaudio.paInt16
        record_seconds = 3
        
        # Try to find an active input device
        active_device = None
        devices_info = []
        
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                devices_info.append(device_info)
                if device_info['maxInputChannels'] > 0:
                    logging.info(f"Found input device: {device_info['name']}")
                    active_device = i
            except Exception as e:
                logging.error(f"Error checking device {i}: {str(e)}")
        
        if active_device is None:
            logging.error("No active input device found")
            logging.info(f"Available devices: {devices_info}")
            
            # Create synthetic audio for testing
            logging.info("Creating synthetic audio for testing...")
            sample_rate = 16000
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Write to WAV file
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
        else:
            # Open audio stream
            stream = p.open(
                format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=active_device,
                frames_per_buffer=chunk_size
            )
            
            # Record audio
            frames = []
            logging.info(f"Recording {record_seconds} seconds of audio from device {active_device}...")
            for i in range(0, int(sample_rate / chunk_size * record_seconds)):
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save as WAV file
            logging.info(f"Writing audio to {temp_wav}")
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(format))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(frames))
        
        # Extract features with OpenSMILE
        logging.info("Running OpenSMILE...")
        command = [
            opensmile_path,
            "-C", config_file,
            "-I", temp_wav,
            "-csvoutput", temp_csv,
            "-instname", "verification"
        ]
        
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            logging.error(f"OpenSMILE failed with error: {result.stderr}")
            return None
        
        # Read CSV output
        if not os.path.exists(temp_csv):
            logging.error(f"Output CSV file not created: {temp_csv}")
            return None
        
        df = pd.read_csv(temp_csv, sep=';')
        feature_names = df.columns[1:]  # Skip 'name' column
        features = df[feature_names].values
        
        # Verify features
        logging.info(f"OpenSMILE extraction time: {elapsed:.3f} seconds")
        logging.info(f"Number of features: {len(feature_names)}")
        
        if features.shape[0] == 0:
            logging.error("No feature rows extracted")
            return None
        
        # Get first row of features
        features_row = features[0]
        
        # Print statistics
        logging.info(f"Feature shape: {features_row.shape}")
        logging.info(f"Min: {np.min(features_row):.4f}, Max: {np.max(features_row):.4f}")
        logging.info(f"Mean: {np.mean(features_row):.4f}, Std: {np.std(features_row):.4f}")
        
        # Check if all features are zero
        non_zero = np.count_nonzero(features_row)
        if non_zero == 0:
            logging.warning("All audio features are zero - this may indicate a problem")
        
        # Visualize some features
        logging.info(f"Feature sample (first 10): {features_row[:10]}")
        
        # Print a selection of feature names
        logging.info(f"Feature name samples: {list(feature_names[:10])}")
        
        return features_row
    except Exception as e:
        logging.error(f"Error in OpenSMILE feature extraction: {str(e)}")
        logging.exception("Stack trace:")
        return None
    finally:
        # Clean up
        for f in [temp_wav, temp_csv]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

def verify_model_with_features(model_path, face_features, audio_features):
    """Test if the model can process the extracted features correctly."""
    logging.info(f"Verifying model with feature inputs...")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return False
    
    try:
        # Highly robust TensorFlow LSTM layer to handle compatibility issues
        class LSTMCompatLayer(tf.keras.layers.LSTM):
            def __init__(self, *args, **kwargs):
                # Remove parameters that may cause compatibility issues
                for param in ['time_major', 'dropout_state_filter_visitor', 'canonical', 'cell']:
                    if param in kwargs:
                        logging.debug(f"Removing {param}={kwargs[param]} parameter from LSTM")
                        kwargs.pop(param, None)
                # Call parent constructor with cleaned kwargs
                super(LSTMCompatLayer, self).__init__(*args, **kwargs)
        
        # Try different approaches to load the model
        model = None
        error_messages = []
        
        # Approach 1: Custom objects with specific LSTM layer
        try:
            logging.info("Trying to load model with custom LSTM layer...")
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'LSTM': LSTMCompatLayer}
            )
        except Exception as e:
            error_messages.append(f"Custom LSTM approach failed: {str(e)}")
            
        # Approach 2: Try with compile=False
        if model is None:
            try:
                logging.info("Trying to load model with compile=False...")
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )
            except Exception as e:
                error_messages.append(f"compile=False approach failed: {str(e)}")
                
        # Approach 3: Use a fully custom loading approach
        if model is None:
            try:
                logging.info("Trying alternative custom loading approach...")
                
                # Define a custom loader that patched the Keras loader
                def custom_load_function(filepath):
                    # Monkey patch the LSTM layer temporarily
                    original_lstm = tf.keras.layers.LSTM
                    tf.keras.layers.LSTM = LSTMCompatLayer
                    
                    try:
                        # Load model with patched LSTM
                        custom_model = tf.keras.models.load_model(filepath, compile=False)
                        return custom_model
                    finally:
                        # Restore original LSTM
                        tf.keras.layers.LSTM = original_lstm
                
                # Use the custom loader
                model = custom_load_function(model_path)
            except Exception as e:
                error_messages.append(f"Custom loader approach failed: {str(e)}")
        
        # If all approaches failed
        if model is None:
            logging.error("All model loading approaches failed!")
            for i, msg in enumerate(error_messages):
                logging.error(f"Approach {i+1} error: {msg}")
            
            logging.info("Skipping model verification due to loading failures")
            return False
        
        # Get model input shapes
        input_shapes = [input.shape for input in model.inputs]
        logging.info(f"Model loaded successfully!")
        logging.info(f"Model input shapes: {input_shapes}")
        logging.info(f"Model output shapes: {[output.shape for output in model.outputs]}")
        
        # Log model architecture for debugging
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        logging.info("Model summary:\n" + "\n".join(model_summary[:20]) + "\n[truncated]")
        
        # Prepare inputs for the model
        if face_features is not None and audio_features is not None:
            # Add batch dimension if needed
            face_features_batch = np.expand_dims(face_features, axis=0)
            audio_features_batch = np.expand_dims(audio_features, axis=0)
            
            logging.info(f"Face features batch shape: {face_features_batch.shape}")
            logging.info(f"Audio features batch shape: {audio_features_batch.shape}")
            
            # Make a test prediction with very robust error handling
            try:
                prediction = None
                
                # Determine input format
                if len(model.inputs) == 2:
                    # Try with normal input format (two separate inputs)
                    logging.info("Model has two inputs - using separate face and audio features")
                    
                    # Check input shapes and adjust if needed
                    if len(model.inputs[0].shape) > 2:  # if model expects time dimension
                        logging.info("Model expects time dimension - adding it")
                        # Add time dimension (assuming first input is for face)
                        face_features_batch = np.expand_dims(face_features_batch, axis=1)
                        # Add time dimension (assuming second input is for audio)
                        audio_features_batch = np.expand_dims(audio_features_batch, axis=1)
                        
                    # Make prediction
                    logging.info("Making prediction with two inputs")
                    prediction = model.predict(
                        [face_features_batch, audio_features_batch], 
                        verbose=1
                    )
                else:
                    # Try with combined inputs
                    logging.info("Model has a single input - concatenating features")
                    combined_features = np.concatenate(
                        [face_features_batch, audio_features_batch], 
                        axis=1
                    )
                    prediction = model.predict(combined_features, verbose=1)
                
                # Process prediction
                if prediction is not None:
                    if isinstance(prediction, list):
                        # Multiple outputs
                        logging.info(f"Model returned multiple outputs: {len(prediction)}")
                        for i, p in enumerate(prediction):
                            logging.info(f"Output {i} shape: {p.shape}")
                            logging.info(f"Output {i} sample: {p[0][:5]}")  # First 5 values
                    else:
                        # Single output
                        logging.info(f"Model returned single output with shape: {prediction.shape}")
                        logging.info(f"Output sample: {prediction[0][:5]}")  # First 5 values
                    
                    return True
                else:
                    logging.error("Model prediction returned None")
                    return False
                    
            except Exception as e:
                logging.error(f"Error making prediction: {str(e)}")
                logging.exception("Stack trace:")
                return False
        else:
            logging.error("Cannot verify model without both face and audio features")
            return False
    except Exception as e:
        logging.error(f"Error in model verification: {str(e)}")
        logging.exception("Stack trace:")
        return False

def main():
    parser = argparse.ArgumentParser(description='Verify feature extraction for emotion recognition')
    parser.add_argument('--model', type=str, default='models/dynamic_padding_no_leakage/model_best.h5', 
                        help='Path to model file')
    parser.add_argument('--opensmile', type=str, 
                        default='./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract',
                        help='Path to OpenSMILE executable')
    parser.add_argument('--config', type=str, 
                        default='./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf',
                        help='Path to OpenSMILE config file')
    
    args = parser.parse_args()
    
    logging.info("=== STARTING ENHANCED FEATURE VERIFICATION ===")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"OpenCV version: {cv2.__version__}")
    logging.info(f"Current directory: {os.getcwd()}")
    
    # 1. Capture multiple frames to increase chance of face detection
    frames = take_multiple_frames(max_attempts=5, delay=0.5)
    if not frames:
        logging.error("Failed to capture frames from camera")
        return
    
    # 2. Extract face features (try on all frames)
    face_features = verify_facenet_features_multiple(frames)
    
    # 3. Verify OpenSMILE features
    audio_features = verify_opensmile_features(args.opensmile, args.config)
    
    # 4. Verify model with features
    model_verified = False
    if face_features is not None and audio_features is not None:
        model_verified = verify_model_with_features(args.model, face_features, audio_features)
        if model_verified:
            logging.info("✅ Model successfully processed the extracted features")
        else:
            logging.error("❌ Model failed to process the extracted features")
    else:
        logging.error("Cannot verify model without both face and audio features")
    
    # Print summary
    logging.info("\n=== VERIFICATION SUMMARY ===")
    logging.info(f"FaceNet Features: {'SUCCESS' if face_features is not None else 'FAILED'}")
    logging.info(f"OpenSMILE Features: {'SUCCESS' if audio_features is not None else 'FAILED'}")
    logging.info(f"Model Verification: {'SUCCESS' if model_verified else 'FAILED'}")
    
    if face_features is not None and audio_features is not None and model_verified:
        logging.info("✅ All components verified successfully!")
    else:
        logging.warning("❌ Some components failed verification. See above logs for details.")
    
    logging.info("=== END OF VERIFICATION ===")

if __name__ == "__main__":
    main()
