#!/usr/bin/env python
"""
Video-Only Emotion Recognition with TensorFlow 2.x Compatible Model

This simplified script provides real-time emotion recognition from webcam video only,
using the TensorFlow 2.x compatible model. It doesn't rely on audio features, which
can cause issues on some systems.

It maintains the same architecture but uses dummy audio features for testing purposes.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
import threading
from collections import deque
import torch
from tensorflow_compatible_model import EmotionRecognitionModel
from facenet_pytorch import MTCNN, InceptionResnetV1
import feature_normalizer

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
FPS = 15  # Target video FPS
DEFAULT_FEATURE_WINDOW = 3  # Default feature window in seconds
FACENET_FEATURE_DIM = 512  # FaceNet feature dimension

class VideoEmotionRecognitionPipeline:
    """
    Video-only real-time emotion recognition pipeline using webcam
    """
    
    def __init__(self, args):
        """
        Initialize the pipeline with the given arguments
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.window_size = args.window_size
        self.feature_window = args.feature_window
        
        # Load the model
        print(f"Loading emotion recognition model from: {args.model}")
        self.model = EmotionRecognitionModel(args.model)
        
        # Setup FaceNet for face detection and feature extraction
        print("Initializing FaceNet for face detection and feature extraction")
        self.face_detector = MTCNN(
            select_largest=True,
            post_process=False,
            device='cpu'
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Setup smoothing - IMPORTANT: Order must match the training data order
        self.predictions_history = deque(maxlen=self.window_size)
        # Original training data used this order: ANG=0, DIS=1, FEA=2, HAP=3, NEU=4, SAD=5
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
        
        # Setup video capture
        self.cap = cv2.VideoCapture(args.camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        self.target_fps = args.fps
        
        # Setup display window
        self.window_name = "Video-Only Emotion Recognition"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, args.display_width, args.display_height)
        
        # Buffer for video frames
        self.video_buffer = np.zeros((0, FACENET_FEATURE_DIM))
        
        # For frame rate calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
    
    def extract_face_features(self, frame):
        """
        Extract face features using FaceNet
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Face embedding features (512-dimensional) or None if no face detected
        """
        # Detect faces
        try:
            # Convert to RGB for FaceNet
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            boxes, probs = self.face_detector.detect(frame_rgb, landmarks=False)
            
            if boxes is not None and len(boxes) > 0:
                # Get the face with highest probability
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                
                # Ensure the box is within the frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                # Extract face
                face = frame_rgb[y1:y2, x1:x2]
                
                # Resize face to match FaceNet input
                face = cv2.resize(face, (160, 160))
                
                # Convert to tensor and normalize
                face = (np.transpose(face, (2, 0, 1)) - 127.5) / 128.0
                face_tensor = torch.from_numpy(face).float().unsqueeze(0)
                
                # Extract features
                with torch.no_grad():
                    embedding = self.facenet(face_tensor)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                return embedding.numpy().flatten(), (x1, y1, x2, y2)
            
            return None, None
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return None, None
    
    def predict_emotion(self):
        """
        Predict emotion using the model
        
        Returns:
            Predicted emotion (string) and probabilities (dict)
        """
        # Check if we have enough features
        if len(self.video_buffer) == 0:
            return "waiting", {}
        
        # Prepare inputs for the model
        video_features = self.video_buffer.reshape(1, -1, FACENET_FEATURE_DIM)
        
        # For audio features - create properly normalized dummy features
        # This matches both the expected input shape AND normalization pattern from training
        seq_length = video_features.shape[1]
        audio_features = feature_normalizer.create_dummy_normalized_features(seq_length)
        
        # Make prediction - ensuring video comes first as in training
        prediction = self.model.predict(video_features, audio_features)
        
        # Get probabilities
        probs = prediction[0]
        
        # Apply smoothing if enabled
        if self.window_size > 1:
            self.predictions_history.append(probs)
            probs = np.mean(np.array(self.predictions_history), axis=0)
        
        # Get the predicted emotion
        emotion_idx = np.argmax(probs)
        emotion = self.emotions[emotion_idx]
        
        # Create probabilities dictionary
        probabilities = {emotion: float(prob) for emotion, prob in zip(self.emotions, probs)}
        
        return emotion, probabilities
    
    def display_results(self, frame, emotion, probabilities, face_box):
        """
        Display the results on the frame
        
        Args:
            frame: Input frame from webcam
            emotion: Predicted emotion
            probabilities: Emotion probabilities
            face_box: Face bounding box
        """
        # Add emotion labels near the face if detected
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            
            # Add emotion label above the face
            label = f"{emotion.upper()}"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
        
        # Create a dark overlay for the emotion bars
        overlay = frame.copy()
        
        # Bar positions and dimensions
        bar_height = 30
        max_bar_width = int(frame.shape[1] * 0.3)
        start_x = 10
        start_y = 50
        
        # Draw header
        cv2.putText(
            overlay, "Emotion Probabilities:",
            (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        
        # Draw bars for each emotion
        for i, (emotion_name, prob) in enumerate(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)):
            # Position
            y = start_y + i * (bar_height + 10)
            
            # Bar width based on probability
            bar_width = int(prob * max_bar_width)
            
            # Get color based on emotion
            if emotion_name == "anger":
                color = (0, 0, 255)  # Red
            elif emotion_name == "disgust":
                color = (0, 128, 128)  # Brown
            elif emotion_name == "fear":
                color = (255, 0, 255)  # Purple
            elif emotion_name == "happiness":
                color = (0, 255, 255)  # Yellow
            elif emotion_name == "sadness":
                color = (255, 0, 0)  # Blue
            else:  # neutral
                color = (128, 128, 128)  # Gray
            
            # Draw bar background
            cv2.rectangle(
                overlay, (start_x, y),
                (start_x + max_bar_width, y + bar_height),
                (50, 50, 50), -1
            )
            
            # Draw filled bar
            cv2.rectangle(
                overlay, (start_x, y),
                (start_x + bar_width, y + bar_height),
                color, -1
            )
            
            # Draw emotion label
            text = f"{emotion_name}: {prob:.2f}"
            cv2.putText(
                overlay, text,
                (start_x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
        
        # Blend overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add FPS information
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time
        
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        
        # Note that this is video-only mode
        cv2.putText(
            frame, "VIDEO-ONLY MODE (dummy audio)",
            (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )
    
    def run(self):
        """
        Run the pipeline
        """
        try:
            print("Starting video-only emotion recognition pipeline")
            print("Press 'q' or 'ESC' to exit")
            
            # Main loop
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                
                # Extract face features
                face_features, face_box = self.extract_face_features(frame)
                
                # If face detected, add features to the buffer
                if face_features is not None:
                    self.video_buffer = np.vstack([self.video_buffer, face_features.reshape(1, -1)])
                    
                    # Keep only the latest features for video (limited to feature window length)
                    max_frames = int(self.target_fps * self.feature_window)
                    if len(self.video_buffer) > max_frames:
                        self.video_buffer = self.video_buffer[-max_frames:]
                
                # Predict emotion
                emotion, probabilities = self.predict_emotion()
                
                # Display results
                self.display_results(frame, emotion, probabilities, face_box)
                
                # Show frame
                cv2.imshow(self.window_name, frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
                # Control frame rate
                time.sleep(max(0, 1/self.target_fps - (time.time() - self.prev_frame_time)))
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            print("Pipeline stopped")

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Video-Only Emotion Recognition with TensorFlow 2.x Compatible Model")
    
    parser.add_argument(
        "--model", type=str,
        default="models/dynamic_padding_no_leakage/model_best.h5",
        help="Path to the model file"
    )
    
    parser.add_argument(
        "--camera_index", type=int,
        default=0,
        help="Camera index"
    )
    
    parser.add_argument(
        "--fps", type=float,
        default=15.0,
        help="Target FPS"
    )
    
    parser.add_argument(
        "--display_width", type=int,
        default=1280,
        help="Display window width"
    )
    
    parser.add_argument(
        "--display_height", type=int,
        default=720,
        help="Display window height"
    )
    
    parser.add_argument(
        "--window_size", type=int,
        default=5,
        help="Window size for smoothing predictions"
    )
    
    parser.add_argument(
        "--feature_window", type=float,
        default=DEFAULT_FEATURE_WINDOW,
        help="Window size in seconds for features"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("Missing dependencies. Please install them with:")
        print("  pip install torch facenet-pytorch")
        sys.exit(1)
    
    # Parse arguments
    args = parse_args()
    
    # Run pipeline
    pipeline = VideoEmotionRecognitionPipeline(args)
    pipeline.run()
