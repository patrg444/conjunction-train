#!/usr/bin/env python3
"""
Verification script for feature extraction in the real-time emotion recognition system.
This script specifically verifies and outputs the features extracted from FaceNet and OpenSMILE
to ensure they are correctly captured and in the expected format for the model.
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

def verify_facenet_features(frame):
    """Extract and verify facial features from a single frame."""
    logging.info("Verifying FaceNet feature extraction...")
    
    try:
        # Initialize FaceNetExtractor
        face_extractor = FaceNetExtractor(
            keep_all=False,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # Extract features
        start_time = time.time()
        features = face_extractor.extract_features(frame)
        elapsed = time.time() - start_time
        
        # Verify features
        if features is None:
            logging.error("FaceNet extraction returned None")
            return False
        
        # Print feature statistics
        logging.info(f"FaceNet feature shape: {features.shape}")
        logging.info(f"Extraction time: {elapsed:.3f} seconds")
        
        non_zero = np.count_nonzero(features)
        total = features.shape[0]
        if non_zero == 0:
            logging.warning("All FaceNet features are zero - no face detected")
        else:
            logging.info(f"Non-zero elements: {non_zero}/{total} ({non_zero/total*100:.1f}%)")
            logging.info(f"Min: {np.min(features):.4f}, Max: {np.max(features):.4f}")
            logging.info(f"Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")
        
        # Visualize some features as a sanity check
        logging.info(f"Feature sample (first 10): {features[:10]}")
        
        return features
    except Exception as e:
        logging.error(f"Error in FaceNet feature extraction: {str(e)}")
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
        
        # Open audio stream
        stream = p.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        # Record audio
        frames = []
        logging.info(f"Recording {record_seconds} seconds of audio...")
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
        
        # Visualize some features
        logging.info(f"Feature sample (first 10): {features_row[:10]}")
        
        # Print a selection of feature names
        logging.info(f"Feature name samples: {list(feature_names[:10])}")
        
        return features_row
    except Exception as e:
        logging.error(f"Error in OpenSMILE feature extraction: {str(e)}")
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
        # Try to load the model with custom LSTM layer to handle parameters
        class LSTMCompatLayer(tf.keras.layers.LSTM):
            def __init__(self, *args, time_major=None, **kwargs):
                if time_major is not None:
                    logging.debug(f"Removing time_major={time_major} parameter from LSTM")
                    kwargs.pop('time_major', None)
                super(LSTMCompatLayer, self).__init__(*args, **kwargs)
        
        logging.info(f"Loading model with custom LSTM layer...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'LSTM': LSTMCompatLayer}
        )
        
        # Get model input shapes
        input_shapes = [input.shape for input in model.inputs]
        logging.info(f"Model input shapes: {input_shapes}")
        logging.info(f"Model output shapes: {[output.shape for output in model.outputs]}")
        
        # Prepare inputs for the model
        if face_features is not None and audio_features is not None:
            # Add batch dimension if needed
            face_features_batch = np.expand_dims(face_features, axis=0)
            audio_features_batch = np.expand_dims(audio_features, axis=0)
            
            logging.info(f"Face features batch shape: {face_features_batch.shape}")
            logging.info(f"Audio features batch shape: {audio_features_batch.shape}")
            
            # Make a test prediction
            try:
                if len(model.inputs) == 2:
                    # Normal case with two separate inputs
                    prediction = model.predict(
                        [face_features_batch, audio_features_batch], 
                        verbose=1
                    )
                else:
                    # Single input model (less common)
                    combined_features = np.concatenate(
                        [face_features_batch, audio_features_batch], 
                        axis=1
                    )
                    prediction = model.predict(combined_features, verbose=1)
                
                # Process prediction
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
    import argparse
    
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
    
    logging.info("=== STARTING FEATURE VERIFICATION ===")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"Current directory: {os.getcwd()}")
    
    # 1. Verify camera and facenet features
    logging.info("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera")
        return
    
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read frame from camera")
        cap.release()
        return
    
    # Save frame for visualization
    cv2.imwrite("verification_frame.jpg", frame)
    logging.info("Frame captured and saved as verification_frame.jpg")
    
    # Extract face features
    face_features = verify_facenet_features(frame)
    
    # Release camera
    cap.release()
    
    # 2. Verify OpenSMILE features
    audio_features = verify_opensmile_features(args.opensmile, args.config)
    
    # 3. Verify model with features
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
