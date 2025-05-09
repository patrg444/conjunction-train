#!/usr/bin/env python3
"""
Demo script for real-time emotion recognition with laughter detection.
This script uses the webcam to detect emotions and laughter in real-time.

Usage:
    python demo_emotion_with_laughter.py [--model MODEL_PATH] [--threshold THRESHOLD]
"""

import os
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyaudio
import threading
import queue
from collections import deque

# Import utilities for audio and video processing
from audio_device_utils import get_best_audio_device
from feature_normalizer import normalize_features, load_normalization_stats

# Set up emotion labels
EMOTION_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo for emotion recognition with laughter detection")
    parser.add_argument("--model", type=str, default="models/audio_pooling_with_laughter/model_best.h5",
                       help="Path to model file")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for laughter detection (0-1)")
    parser.add_argument("--max_seq_len", type=int, default=45,
                       help="Maximum sequence length for input")
    parser.add_argument("--fps", type=int, default=15,
                       help="Frames per second to process")
    parser.add_argument("--no_audio", action="store_true",
                       help="Run without audio input (video only)")
    return parser.parse_args()


class AudioProcessor:
    """Audio processor for real-time audio feature extraction."""
    
    def __init__(self, max_seq_len=45, no_audio=False):
        """Initialize audio processor."""
        self.max_seq_len = max_seq_len
        self.no_audio = no_audio
        self.audio_queue = queue.Queue()
        self.audio_features = deque(maxlen=max_seq_len)
        
        # Fill with zeros initially
        for _ in range(max_seq_len):
            self.audio_features.append(np.zeros(88))
        
        # Setup audio if needed
        if not no_audio:
            self.stream = None
            self.p = pyaudio.PyAudio()
            self.setup_audio()
            self.audio_thread = threading.Thread(target=self.process_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
    
    def setup_audio(self):
        """Set up audio stream."""
        device_id = get_best_audio_device(self.p)
        if device_id is None:
            print("Warning: No suitable audio device found!")
            self.no_audio = True
            return
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=512,
            input_device_index=device_id,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio data."""
        if self.no_audio:
            return
        
        try:
            import opensmile
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                num_channels=1,
                sample_rate=16000
            )
            
            while True:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    audio_np = np.frombuffer(audio_data, dtype=np.float32)
                    
                    # Extract features using OpenSmile
                    features = smile.process_signal(audio_np, 16000)
                    features_np = features.values
                    
                    # Add features to the deque
                    for i in range(features_np.shape[0]):
                        self.audio_features.append(features_np[i])
                
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            self.no_audio = True
    
    def get_features(self):
        """Get current audio features."""
        if self.no_audio:
            # Return zeros if no audio
            return np.zeros((self.max_seq_len, 88))
        
        # Convert deque to numpy array
        return np.array(list(self.audio_features))
    
    def cleanup(self):
        """Clean up audio resources."""
        if not self.no_audio and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()


class VideoProcessor:
    """Video processor for real-time face feature extraction."""
    
    def __init__(self, max_seq_len=45):
        """Initialize video processor."""
        self.max_seq_len = max_seq_len
        self.video_features = deque(maxlen=max_seq_len)
        
        # Fill with zeros initially
        for _ in range(max_seq_len):
            self.video_features.append(np.zeros(512))
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load FaceNet model for facial embeddings
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.mtcnn = MTCNN(keep_all=False, device='cpu')
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        except ImportError:
            print("Warning: facenet_pytorch not installed. Using dummy embeddings.")
            self.mtcnn = None
            self.facenet = None
    
    def process_frame(self, frame):
        """Process video frame to extract facial embeddings."""
        try:
            # Convert to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.mtcnn is not None and self.facenet is not None:
                # Detect face and get embedding
                import torch
                img = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()
                img /= 255.0
                
                # Detect faces
                boxes, _ = self.mtcnn.detect(img)
                
                if boxes is not None and len(boxes) > 0:
                    # Get largest face
                    box = boxes[0]
                    
                    # Draw box on frame
                    cv2.rectangle(frame, 
                                 (int(box[0]), int(box[1])), 
                                 (int(box[2]), int(box[3])), 
                                 (0, 255, 0), 2)
                    
                    # Extract face embedding
                    face = self.mtcnn(img)
                    if face is not None:
                        embedding = self.facenet(face.unsqueeze(0))
                        self.video_features.append(embedding.detach().numpy().flatten())
                    else:
                        # Use last embedding if face detection failed
                        self.video_features.append(self.video_features[-1])
                else:
                    # Use last embedding if no face detected
                    self.video_features.append(self.video_features[-1])
            else:
                # Use dummy embeddings
                self.video_features.append(np.random.normal(0, 0.1, 512))
        except Exception as e:
            print(f"Video processing error: {e}")
            # Use last embedding if error
            self.video_features.append(self.video_features[-1])
        
        return frame
    
    def get_features(self):
        """Get current video features."""
        return np.array(list(self.video_features))


def predict_emotion_with_laughter(model, audio_features, video_features, threshold=0.5):
    """
    Predict emotion and laughter using the model.
    
    Args:
        model: Trained model with emotion and laughter outputs
        audio_features: Audio features (max_seq_len, 88)
        video_features: Video features (max_seq_len, 512)
        threshold: Threshold for laughter detection (0-1)
        
    Returns:
        emotion_label: Predicted emotion label
        emotion_prob: Probability of predicted emotion
        is_laughing: Boolean indicating if laughter detected
        laugh_prob: Probability of laughter
    """
    try:
        # Normalize features
        norm_audio = normalize_features(audio_features, name="audio")
        norm_video = normalize_features(video_features, name="video")
        
        # Expand dimensions for batch size
        norm_audio = np.expand_dims(norm_audio, axis=0)
        norm_video = np.expand_dims(norm_video, axis=0)
        
        # Make prediction
        emotion_probs, laugh_prob = model.predict([norm_audio, norm_video], verbose=0)
        
        # Get highest probability emotion
        emotion_idx = np.argmax(emotion_probs[0])
        emotion_label = EMOTION_LABELS.get(emotion_idx, "unknown")
        emotion_prob = emotion_probs[0][emotion_idx]
        
        # Get laughter probability
        laugh_prob = float(laugh_prob[0][0])
        is_laughing = laugh_prob >= threshold
        
        return emotion_label, emotion_prob, is_laughing, laugh_prob
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown", 0.0, False, 0.0


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load model
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    model.summary()
    
    # Initialize processors
    audio_processor = AudioProcessor(max_seq_len=args.max_seq_len, no_audio=args.no_audio)
    video_processor = VideoProcessor(max_seq_len=args.max_seq_len)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Frame timing for FPS control
    frame_time = 1.0 / args.fps
    last_frame_time = time.time()
    
    try:
        while True:
            # Control FPS
            current_time = time.time()
            delta = current_time - last_frame_time
            
            if delta < frame_time:
                time.sleep(frame_time - delta)
            
            last_frame_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read from webcam")
                break
            
            # Process frame to extract facial features
            frame = video_processor.process_frame(frame)
            
            # Get current features
            audio_features = audio_processor.get_features()
            video_features = video_processor.get_features()
            
            # Make prediction
            emotion, emotion_prob, is_laughing, laugh_prob = predict_emotion_with_laughter(
                model, audio_features, video_features, threshold=args.threshold
            )
            
            # Display prediction on frame
            # Create background rectangle for text
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            
            # Display emotion
            text = f"Emotion: {emotion} ({emotion_prob:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display laughter
            if is_laughing:
                text = f"Laughter detected! ({laugh_prob:.2f})"
                cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                text = f"No laughter ({laugh_prob:.2f})"
                cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Emotion Recognition with Laughter Detection", frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        audio_processor.cleanup()
        print("Demo completed.")


if __name__ == "__main__":
    main()
