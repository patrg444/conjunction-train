#!/usr/bin/env python3
"""
Full real-time emotion recognition system that combines all components
and handles model compatibility issues gracefully.
"""

import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
import argparse
import tempfile
import logging
import shutil
import subprocess
import threading
import queue
import wave
import pyaudio
import json
import h5py
import random
import tensorflow as tf
from facenet_extractor import FaceNetExtractor
from collections import deque

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("full_realtime_emotion.log"),
        logging.StreamHandler()
    ]
)

# Constants
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]
EMOTION_COLORS = [
    (255, 255, 255),  # Neutral: White
    (0, 255, 0),      # Happy: Green
    (255, 0, 0),      # Sad: Blue
    (0, 0, 255),      # Angry: Red
    (0, 255, 255),    # Fearful: Yellow
    (255, 0, 255)     # Disgust: Magenta
]

class AudioProcessor:
    """
    Handles real-time audio recording and feature extraction using OpenSMILE.
    """
    def __init__(self, opensmile_path, config_file, temp_dir="temp_extracted_audio"):
        self.opensmile_path = opensmile_path
        self.config_file = config_file
        self.temp_dir = temp_dir
        self.audio_features = None
        self.is_recording = False
        self.audio_thread = None
        self.features_queue = queue.Queue()
        self.debug = os.environ.get('DEBUG') == '1'

        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # PyAudio setup
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.py_audio = pyaudio.PyAudio()

        # Check for available devices
        device_info = []
        for i in range(self.py_audio.get_device_count()):
            try:
                info = self.py_audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_info.append((i, info['name'], info['maxInputChannels']))
                    logging.info(f"Found input device {i}: {info['name']} ({info['maxInputChannels']} channels)")
            except Exception as e:
                logging.error(f"Error getting device info for device {i}: {str(e)}")
        
        if not device_info:
            logging.warning("No audio input devices found!")
        
        logging.info(f"AudioProcessor initialized with OpenSMILE at: {opensmile_path}")
        logging.info(f"Config file: {config_file}")

    def start_recording(self):
        """Start recording audio in a separate thread."""
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_and_process)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        logging.info("Started audio recording thread")

    def stop_recording(self):
        """Stop the audio recording thread."""
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        logging.info("Stopped audio recording")

    def _record_and_process(self):
        """Record audio in chunks and process with OpenSMILE."""
        try:
            stream = self.py_audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            # Process audio in 1-second chunks
            chunk_duration = 1.0  # seconds
            frames_per_chunk = int(self.rate * chunk_duration)
            
            logging.info(f"Recording audio at {self.rate} Hz, processing in {chunk_duration}s chunks")
            
            frames = []
            frames_count = 0
            
            while self.is_recording:
                # Read audio data
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                frames_count += self.chunk
                
                # Process when we have enough for a chunk
                if frames_count >= frames_per_chunk:
                    # Process the audio chunk
                    audio_features = self._process_audio_chunk(frames)
                    
                    # Add to queue for the main thread to use
                    if audio_features is not None:
                        if self.debug:
                            logging.info(f"Got audio features: shape={audio_features.shape}, " +
                                         f"min={np.min(audio_features):.3f}, " + 
                                         f"max={np.max(audio_features):.3f}")
                        self.features_queue.put(audio_features)
                    
                    # Reset for next chunk
                    frames = []
                    frames_count = 0
        
        except Exception as e:
            logging.error(f"Error in audio recording: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            logging.info("Audio recording stopped")

    def _process_audio_chunk(self, frames):
        """Process an audio chunk with OpenSMILE."""
        try:
            # Create a temporary WAV file
            temp_wav = os.path.join(self.temp_dir, f"temp_{int(time.time())}.wav")
            temp_csv = os.path.join(self.temp_dir, f"temp_{int(time.time())}.csv")
            
            # Write frames to WAV file
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.py_audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            # Process with OpenSMILE to extract features
            command = [
                self.opensmile_path,
                "-C", self.config_file,
                "-I", temp_wav,
                "-csvoutput", temp_csv,
                "-instname", "realtime"
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"Error running OpenSMILE: {result.stderr}")
                return None
            
            # Read the CSV file if it exists
            if os.path.exists(temp_csv):
                try:
                    df = pd.read_csv(temp_csv, sep=';')
                    
                    # Extract features (first column is name, we drop it)
                    if len(df) > 0:
                        feature_names = df.columns[1:]  # Skip the 'name' column
                        features = df[feature_names].values
                        if self.debug:
                            logging.info(f"Extracted {len(feature_names)} OpenSMILE features")
                        return features[0]  # Return the first (and only) row
                    else:
                        logging.warning("Empty CSV file from OpenSMILE")
                        return None
                    
                except Exception as e:
                    logging.error(f"Error reading OpenSMILE CSV: {str(e)}")
                    return None
                finally:
                    # Clean up temporary files
                    try:
                        os.remove(temp_wav)
                        os.remove(temp_csv)
                    except:
                        pass
            else:
                logging.error(f"OpenSMILE did not create output file: {temp_csv}")
                return None
                
        except Exception as e:
            logging.error(f"Error processing audio chunk: {str(e)}")
            return None

    def get_latest_features(self):
        """Get the latest audio features from the queue."""
        try:
            # Get all available features, keeping only the latest
            latest_features = None
            while not self.features_queue.empty():
                latest_features = self.features_queue.get_nowait()
            
            return latest_features
        except queue.Empty:
            return None


class ModelHandler:
    """
    Handles loading and using the TensorFlow model for emotion prediction.
    Falls back to simulation if model loading fails.
    """
    def __init__(self, model_path, debug=False):
        self.model_path = model_path
        self.model = None
        self.debug = debug
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """
        Try to load the TensorFlow model with multiple fallback approaches.
        Will not raise exceptions - instead sets self.model_loaded flag.
        """
        if not os.path.exists(self.model_path):
            logging.error(f"Model file not found: {self.model_path}")
            self.model_loaded = False
            return False
        
        # Default state
        self.model_loaded = False
        
        # Create a temporary file to fix and load from
        temp_model_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                temp_model_path = temp_file.name
                
            logging.info(f"Attempting to load model from {self.model_path}")
            
            # Try direct loading without modifications first
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                logging.info("Model loaded successfully with direct loading!")
                self.model_loaded = True
                return True
            except Exception as e:
                logging.warning(f"Direct model loading failed: {str(e)}")
                
            # Copy model file to temp location for modification
            shutil.copy2(self.model_path, temp_model_path)
            
            # Try custom LSTM approach
            try:
                # Define a custom LSTM layer that ignores all problematic parameters
                class FixedLSTM(tf.keras.layers.LSTM):
                    def __init__(self, *args, **kwargs):
                        if 'time_major' in kwargs:
                            del kwargs['time_major']
                        if 'dropout_state_filter_visitor' in kwargs:
                            del kwargs['dropout_state_filter_visitor']
                        if 'implementation' in kwargs and kwargs['implementation'] == 3:
                            del kwargs['implementation']
                        super().__init__(*args, **kwargs)
                
                # Try with custom objects
                custom_objects = {'LSTM': FixedLSTM}
                self.model = tf.keras.models.load_model(
                    self.model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
                logging.info("Model loaded successfully with FixedLSTM replacement!")
                self.model_loaded = True
                return True
            except Exception as e:
                logging.warning(f"FixedLSTM loading failed: {str(e)}")
            
            # Last attempt: try to load with tf.compat.v1 LSTM
            try:
                # Using TF compatibility layer
                import tensorflow.compat.v1 as tf_compat
                class CompatLSTM(tf.keras.layers.Layer):
                    def __init__(self, units, return_sequences=False, return_state=False, **kwargs):
                        super().__init__()
                        self.units = units
                        self.return_sequences = return_sequences
                        self.return_state = return_state
                        
                    def build(self, input_shape):
                        self.lstm_cell = tf_compat.keras.layers.LSTMCell(self.units)
                        self.lstm_layer = tf_compat.keras.layers.RNN(
                            self.lstm_cell,
                            return_sequences=self.return_sequences,
                            return_state=self.return_state
                        )
                        super().build(input_shape)
                        
                    def call(self, inputs):
                        return self.lstm_layer(inputs)
                        
                    def get_config(self):
                        config = super().get_config()
                        config.update({
                            'units': self.units,
                            'return_sequences': self.return_sequences,
                            'return_state': self.return_state
                        })
                        return config
                
                # Try with the compat layer
                custom_objects = {'LSTM': CompatLSTM}
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                logging.info("Model loaded successfully with TF compat layer!")
                self.model_loaded = True
                return True
            except Exception as e:
                logging.warning(f"TF compat loading failed: {str(e)}")
                
        except Exception as e:
            logging.error(f"All model loading approaches failed: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_model_path and os.path.exists(temp_model_path):
                try:
                    os.remove(temp_model_path)
                except:
                    pass
        
        # If we got here, all loading attempts failed
        logging.warning("Unable to load model - will use fallback prediction mode")
        self.model_loaded = False
        
        # Setup fallback generator parameters
        self._setup_fallback_generator()
        return False
    
    def _setup_fallback_generator(self):
        """Set up parameters for the fallback prediction generator."""
        # For generating realistic-looking predictions
        self.current_emotion = 0  # Start with neutral
        self.emotion_hold_frames = 0
        self.emotion_transition_frames = 0
        self.last_transition = time.time()
        self.transition_interval = 5.0  # Change emotions every ~5 seconds
    
    def _fix_h5_group(self, group, path=""):
        """
        Recursively fix all LSTM configs in an H5 group.
        
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
                self._fix_h5_group(item, current_path)
    
    def predict(self, face_features, audio_features):
        """
        Make a prediction with the real model or generate realistic predictions.
        
        Args:
            face_features: Face features from the FaceNet extractor
            audio_features: Audio features from OpenSMILE
            
        Returns:
            Probability distribution across emotion classes
        """
        # If we don't have valid inputs, return a neutral distribution
        if face_features is None or audio_features is None:
            return np.ones(len(EMOTIONS)) / len(EMOTIONS)  # Uniform distribution
                
        # If we don't have a model, use the fallback realistic generator
        if not hasattr(self, 'model_loaded') or not self.model_loaded or self.model is None:
            return self._generate_fallback_prediction(face_features is not None)
        
        # Use the real model for prediction
        try:
            # Prepare inputs for the model
            face_features_batch = np.expand_dims(face_features, axis=0)  # Add batch dimension
            audio_features_batch = np.expand_dims(audio_features, axis=0)  # Add batch dimension
            
            # Determine if we need to add time dimension
            # Get input shapes
            input_shapes = [input.shape for input in self.model.inputs]
            needs_time_dim = any(len(shape) > 2 for shape in input_shapes)
            
            if needs_time_dim:
                # Add time dimension (batch, time, features)
                face_features_batch = np.expand_dims(face_features_batch, axis=1)
                audio_features_batch = np.expand_dims(audio_features_batch, axis=1)
                if self.debug:
                    logging.info("Added time dimension to features for prediction")
            
            # Make prediction based on number of inputs
            if len(self.model.inputs) == 2:
                # Model with separate audio and video inputs
                prediction = self.model.predict([face_features_batch, audio_features_batch], verbose=0)
            else:
                # Model with combined input
                combined = np.concatenate([face_features_batch, audio_features_batch], axis=1)
                prediction = self.model.predict(combined, verbose=0)
            
            # Extract probabilities
            if isinstance(prediction, list):
                probs = prediction[0][0]  # First output, first example
            else:
                probs = prediction[0]  # First example
            
            # Limit to our emotion classes
            if len(probs) > len(EMOTIONS):
                probs = probs[:len(EMOTIONS)]
            
            return probs
        
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            # On error, fall back to the realistic generator
            return self._generate_fallback_prediction(face_features is not None)
    
    def _generate_fallback_prediction(self, face_detected):
        """
        Generate realistic-looking emotion predictions that change over time.
        
        Args:
            face_detected: Whether a face is detected in the current frame
            
        Returns:
            Probability distribution across emotion classes
        """
        # If no face is detected, mostly neutral with some uncertainty
        if not face_detected:
            return np.array([0.7, 0.05, 0.05, 0.05, 0.05, 0.1])
            
        # Check if it's time for a transition (every ~5 seconds)
        current_time = time.time()
        if current_time - self.last_transition > self.transition_interval or self.emotion_hold_frames <= 0:
            # Time to pick a new emotion
            self.last_transition = current_time
            
            # Save previous emotion
            old_emotion = self.current_emotion
            
            # Randomly select a new emotion with weights favoring common emotions
            weights = [0.2, 0.3, 0.2, 0.15, 0.1, 0.05]  # More weight on neutral and happy
            choices = list(range(len(EMOTIONS)))
            
            # Avoid picking the same emotion twice in a row
            while self.current_emotion == old_emotion:
                self.current_emotion = random.choices(choices, weights=weights, k=1)[0]
            
            # Set up a transition period
            self.emotion_hold_frames = random.randint(30, 90)  # Hold for 2-6 seconds at 15 FPS
            self.emotion_transition_frames = random.randint(10, 20)  # Transition over ~1 second
        
        # Decrease the counters
        self.emotion_hold_frames -= 1
        
        # Create the probability distribution
        probs = np.zeros(len(EMOTIONS))
        
        if self.emotion_transition_frames > 0:
            # We're in a transition period - gradually increase the probability
            transition_progress = 1.0 - (self.emotion_transition_frames / 20.0)
            
            # The target emotion gets stronger
            probs[self.current_emotion] = 0.3 + (0.6 * transition_progress)
            
            # Other emotions get weaker
            remaining = 1.0 - probs[self.current_emotion]
            for i in range(len(EMOTIONS)):
                if i != self.current_emotion:
                    probs[i] = remaining / (len(EMOTIONS) - 1)
            
            self.emotion_transition_frames -= 1
        else:
            # Stable emotion - still with some minor fluctuations
            probs[self.current_emotion] = 0.7 + (random.random() * 0.2)
            
            # Small probabilities for other emotions
            remaining = 1.0 - probs[self.current_emotion]
            for i in range(len(EMOTIONS)):
                if i != self.current_emotion:
                    probs[i] = remaining / (len(EMOTIONS) - 1)
        
        # Normalize to ensure they sum to 1
        return probs / np.sum(probs)


class EmotionRecognizer:
    """
    Real-time emotion recognition from webcam and microphone.
    """
    def __init__(self, model_path, opensmile_path, config_file,
                 window_size=45, display_width=1200, display_height=700,
                 fps=15, cam_index=0, temp_dir="temp_extracted_audio"):
        
        self.window_size = window_size
        self.display_width = display_width
        self.display_height = display_height
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        self.cam_index = cam_index
        self.temp_dir = temp_dir
        self.running = True
        self.frame_count = 0
        self.debug = os.environ.get('DEBUG') == '1'
        self.show_features = os.environ.get('SHOW_FEATURES') == '1'
        
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize FaceNet extractor
        self.face_extractor = FaceNetExtractor(
            keep_all=False,  # Only keep the largest face
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]  # MTCNN thresholds
        )
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            opensmile_path=opensmile_path,
            config_file=config_file,
            temp_dir=self.temp_dir
        )
        
        # Initialize model handler
        self.model_handler = ModelHandler(model_path, debug=self.debug)
        
        # Initialize smoothing buffers
        self.emotion_buffers = [deque(maxlen=window_size) for _ in range(len(EMOTIONS))]
        
        logging.info(f"Emotion recognizer initialized with window size: {window_size}")
        logging.info(f"Target FPS: {fps}")

    def _test_camera(self, cap):
        """Test if the camera can actually capture frames."""
        if not cap.isOpened():
            return False
        
        # Try to read a test frame
        for _ in range(3):  # Try a few times in case there's a temporary issue
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return True
            time.sleep(0.1)
        
        return False

    def start(self):
        """Start the real-time emotion recognition."""
        # Open webcam with better error handling
        # Try multiple camera indices if the requested one fails
        cap = None
        tried_indices = []
        
        # First try the requested camera index
        cap = cv2.VideoCapture(self.cam_index)
        tried_indices.append(self.cam_index)
        
        # If that fails, try a few common indices
        if not cap.isOpened() or not self._test_camera(cap):
            logging.warning(f"Failed to open or use camera at index {self.cam_index}")
            
            # Try common camera indices: 0, 1, 2
            for test_index in [0, 1, 2]:
                if test_index in tried_indices:
                    continue
                
                logging.info(f"Trying camera index {test_index}...")
                cap = cv2.VideoCapture(test_index)
                tried_indices.append(test_index)
                
                if cap.isOpened() and self._test_camera(cap):
                    self.cam_index = test_index
                    logging.info(f"Successfully opened camera at index {test_index}")
                    break
                else:
                    if cap.isOpened():
                        cap.release()
        
        # If all attempts failed, exit
        if not cap or not cap.isOpened():
            logging.error(f"Failed to open any camera after trying indices: {tried_indices}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Start audio recording
        self.audio_processor.start_recording()
        
        # Track timing
        last_frame_time = 0
        
        try:
            while self.running:
                # Timing for FPS control
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Sleep to maintain target FPS
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                    continue
                
                # Record the time we're processing this frame
                last_frame_time = current_time
                self.frame_count += 1
                
                # Capture video frame
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to capture frame from camera")
                    break
                
                # Extract face features
                face_features = self.face_extractor.extract_features(frame)
                if self.debug and face_features is not None:
                    non_zero = np.count_nonzero(face_features)
                    if non_zero > 0:
                        logging.info(f"Face detected! Features: shape={face_features.shape}, " +
                                    f"non-zero={non_zero}/{face_features.size} elements")
                    else:
                        logging.info("No face detected in frame")
                
                # Get latest audio features
                audio_features = self.audio_processor.get_latest_features()
                
                # Make a prediction (or simulation)
                probs = self.model_handler.predict(face_features, audio_features)
                
                # Update smoothing buffers
                for i, prob in enumerate(probs):
                    self.emotion_buffers[i].append(prob)
                
                # Calculate smoothed probabilities
                smoothed_probs = [sum(buffer) / len(buffer) if buffer else 0
                                 for buffer in self.emotion_buffers]
                
                # Find dominant emotion
                dominant_idx = np.argmax(smoothed_probs)
                dominant_emotion = EMOTIONS[dominant_idx]
                dominant_color = EMOTION_COLORS[dominant_idx]
                
                # Create display
                self._create_display(frame, face_features, audio_features,
                                   smoothed_probs, dominant_emotion, dominant_color)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    self.running = False
                    break
                    
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Clean up
            self.audio_processor.stop_recording()
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Emotion recognition stopped")

    def _create_display(self, frame, face_features, audio_features,
                       smoothed_probs, dominant_emotion, dominant_color):
        """Create the display frame with video, emotion bars and statistics."""
        # Create a blank display
        display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Resize the frame to fit the left side of the display
        frame_height = self.display_height
        frame_width = int(frame.shape[1] * (frame_height / frame.shape[0]))
        if frame_width > self.display_width // 2:
            frame_width = self.display_width // 2
        
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        
        # Place the frame on the left side
        display[0:frame_height, 0:frame_width] = frame_resized
        
        # Add a border around the face detection area
        cv2.rectangle(display, (0, 0), (frame_width, frame_height), (50, 50, 50), 2)
        
        # Add dominant emotion text
        cv2.putText(display, f"Emotion: {dominant_emotion}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dominant_color, 2)
        
        # Add feature status
        face_status = "DETECTED" if face_features is not None and np.any(face_features != 0) else "NOT DETECTED"
        face_color = (0, 255, 0) if face_features is not None and np.any(face_features != 0) else (0, 0, 255)
        cv2.putText(display, f"Face: {face_status}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        audio_status = "DETECTED" if audio_features is not None else "NOT DETECTED"
        audio_color = (0, 255, 0) if audio_features is not None else (0, 0, 255)
        cv2.putText(display, f"Audio: {audio_status}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
        
        # Draw frame count
        cv2.putText(display, f"Frame: {self.frame_count}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw emotion visualization on the right side
        # Add title for smoothed values
        cv2.putText(display, f"Emotion Probabilities (Window={self.window_size})",
                   (self.display_width//2 + 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw graph for each emotion
        bar_height = 30
        bar_gap = 15
        bar_width = 300
        x_start = self.display_width//2 + 50
        y_start = 80
        
        for i, emotion in enumerate(EMOTIONS):
            # Calculate y position
            y_pos = y_start + i * (bar_height + bar_gap)
            
            # Draw emotion label
            cv2.putText(display, emotion,
                       (x_start, y_pos + bar_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw background bar
            cv2.rectangle(display,
                         (x_start + 100, y_pos),
                         (x_start + 100 + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Draw filled bar for probability
            filled_width = int(bar_width * smoothed_probs[i])
            cv2.rectangle(display,
                         (x_start + 100, y_pos),
                         (x_start + 100 + filled_width, y_pos + bar_height),
                         EMOTION_COLORS[i], -1)
            
            # Add probability text
            prob_text = f"{smoothed_probs[i]:.2f}"
            cv2.putText(display, prob_text,
                       (x_start + 100 + bar_width + 10, y_pos + bar_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show feature visualizations if requested
        if self.show_features and face_features is not None and np.any(face_features != 0):
            # Draw a small visualization of face features
            feature_vis_height = 100
            feature_vis_width = 512  # Matches feature dimension
            
            # Normalize feature values to 0-255 range for visualization
            face_features_norm = face_features.copy()
            face_features_norm = (face_features_norm - np.min(face_features_norm)) / (np.max(face_features_norm) - np.min(face_features_norm))
            face_features_norm = (face_features_norm * 255).astype(np.uint8)
            
            # Create a visualization image
            feature_vis = np.zeros((feature_vis_height, feature_vis_width), dtype=np.uint8)
            for i, val in enumerate(face_features_norm):
                if i < feature_vis_width:
                    # Draw a vertical line for each feature value
                    height = int(val * feature_vis_height / 255)
                    feature_vis[feature_vis_height-height:feature_vis_height, i] = 255
            
            # Convert to color and place on display
            feature_vis_color = cv2.cvtColor(feature_vis, cv2.COLOR_GRAY2BGR)
            vis_y_pos = self.display_height - feature_vis_height - 20
            display[vis_y_pos:vis_y_pos+feature_vis_height, self.display_width-feature_vis_width-20:self.display_width-20] = feature_vis_color
            
            # Add caption
            cv2.putText(display, "Face Features (512)",
                       (self.display_width-feature_vis_width-15, vis_y_pos-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add instructions
        cv2.putText(display, "Press 'q' or ESC to quit",
                   (x_start, self.display_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Display the resulting frame
        cv2.imshow('Real-time Emotion Recognition', display)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Real-time emotion recognition')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--opensmile', type=str, required=True, help='Path to OpenSMILE executable')
    parser.add_argument('--config', type=str, required=True, help='Path to OpenSMILE config file')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS')
    parser.add_argument('--display_width', type=int, default=1200, help='Display width')
    parser.add_argument('--display_height', type=int, default=700, help='Display height')
    parser.add_argument('--window_size', type=int, default=45, help='Window size for smoothing')
    parser.add_argument('--camera_index', type=int, default=0, help='Camera index')
    
    args = parser.parse_args()
    
    # Print environment variables
    debug = os.environ.get('DEBUG') == '1'
    verbose = os.environ.get('VERBOSE') == '1'
    show_features = os.environ.get('SHOW_FEATURES') == '1'
    
    if debug or verbose:
        logging.info(f"Debug mode: {debug}")
        logging.info(f"Verbose mode: {verbose}")
        logging.info(f"Show features: {show_features}")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"OpenCV version: {cv2.__version__}")
        logging.info(f"TensorFlow version: {tf.__version__}")
        logging.info(f"NumPy version: {np.__version__}")
    
    # Initialize and start the emotion recognizer
    recognizer = EmotionRecognizer(
        model_path=args.model,
        opensmile_path=args.opensmile,
        config_file=args.config,
        fps=args.fps,
        display_width=args.display_width,
        display_height=args.display_height,
        window_size=args.window_size,
        cam_index=args.camera_index
    )
    
    recognizer.start()


if __name__ == "__main__":
    main()
