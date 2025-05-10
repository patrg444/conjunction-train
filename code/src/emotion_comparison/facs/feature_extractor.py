#!/usr/bin/env python3
# coding: utf-8
"""
FACS Feature Extractor for emotion recognition.
This module implements facial action coding system (FACS) based feature extraction.
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import dlib
from pathlib import Path

# Check if OpenFace is installed
OPENFACE_INSTALLED = False
try:
    import openface
    OPENFACE_INSTALLED = True
except ImportError:
    print("OpenFace Python bindings not installed. Using fallback methods.")

class FACSFeatureExtractor:
    """
    Extracts Facial Action Coding System (FACS) features from facial videos.

    FACS features are based on the movement of specific facial muscles which are
    categorized into Action Units (AUs). These AUs are the fundamental actions of
    individual muscles or groups of muscles and are particularly useful for
    emotion recognition.
    """

    def __init__(self, feature_dim=128, au_model_path=None,
                 face_detector_path=None, landmark_predictor_path=None):
        """
        Initialize the FACS feature extractor.

        Args:
            feature_dim: Dimension of the extracted features
            au_model_path: Path to pre-trained AU detection model (optional)
            face_detector_path: Path to face detector model (optional)
            landmark_predictor_path: Path to facial landmark predictor model (optional)
        """
        self.feature_dim = feature_dim
        self.au_model_path = au_model_path

        # Initialize face detector
        self.face_detector = None
        self.landmark_predictor = None

        # Try to use dlib's face detector and predictor if paths are provided
        if face_detector_path and os.path.exists(face_detector_path):
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                print("Using dlib face detector")
            except Exception as e:
                print(f"Error initializing dlib face detector: {e}")

        if landmark_predictor_path and os.path.exists(landmark_predictor_path):
            try:
                self.landmark_predictor = dlib.shape_predictor(landmark_predictor_path)
                print(f"Loaded landmark predictor from {landmark_predictor_path}")
            except Exception as e:
                print(f"Error loading landmark predictor: {e}")

        # If dlib models are not available use OpenCV's face detector
        if self.face_detector is None:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Using OpenCV face detector")

        # Initialize OpenFace if available
        self.openface_aligner = None
        if OPENFACE_INSTALLED:
            try:
                # Initialize OpenFace alignment and AU detection
                self.openface_aligner = openface.AlignDlib(landmark_predictor_path)
                print("OpenFace initialized successfully")
            except Exception as e:
                print(f"Error initializing OpenFace: {e}")

        # If feature_dim > 0 we'll use a dimensionality reduction approach
        self.use_dimensionality_reduction = feature_dim > 0

    def extract_features(self, video_path, save_visualization=False,
                         output_dir=None, max_frames=None):
        """
        Extract FACS features from a video file.

        Args:
            video_path: Path to video file
            save_visualization: Whether to save visualization of detected AUs
            output_dir: Directory to save visualizations
            max_frames: Maximum number of frames to process

        Returns:
            features: Extracted features (1D numpy array)
        """
        # Get frames from video
        frames = self._extract_frames_from_video(video_path, max_frames)

        if frames is None or len(frames) == 0:
            print(f"No frames extracted from {video_path}")
            return None

        # Extract AUs for each frame
        all_aus = []
        for i, frame in enumerate(tqdm(frames, desc="Extracting AUs", leave=False)):
            aus = self._extract_aus_from_frame(frame)
            if aus is not None:
                all_aus.append(aus)

            # Save visualization for some frames if requested
            if save_visualization and output_dir and i % 10 == 0:
                self._visualize_aus(frame, aus, os.path.basename(video_path),
                                  i, output_dir)

        if not all_aus:
            print(f"No AUs extracted from {video_path}")
            return None

        # Convert to numpy array
        all_aus = np.array(all_aus)

        # Extract temporal statistics or reduce dimensionality
        if self.use_dimensionality_reduction:
            features = self._reduce_dimensionality(all_aus)
        else:
            features = self._extract_temporal_statistics(all_aus)

        return features

    def _extract_frames_from_video(self, video_path, max_frames=None):
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract

        Returns:
            frames: List of extracted frames
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            if max_frames and len(frames) >= max_frames:
                break

        cap.release()

        if len(frames) == 0:
            print(f"No frames extracted from {video_path}")
            return None

        return frames

    def _extract_aus_from_frame(self, frame):
        """
        Extract Action Units from a frame.

        Args:
            frame: Input frame

        Returns:
            aus: Dictionary of Action Units and their intensities
        """
        # If OpenFace is available use it for AU extraction
        if OPENFACE_INSTALLED and self.openface_aligner:
            return self._extract_aus_openface(frame)
        else:
            # Fallback to our own AU extraction method
            return self._extract_aus_fallback(frame)

    def _extract_aus_openface(self, frame):
        """
        Extract AUs using OpenFace.

        Args:
            frame: Input frame

        Returns:
            aus: Dictionary of Action Units and their intensities
        """
        try:
            # Detect face using OpenFace
            bb = self.openface_aligner.getLargestFaceBoundingBox(frame)
            if bb is None:
                return None

            # Extract AUs using OpenFace
            # Note: This is a placeholder - actual OpenFace API calls would go here
            aus = {}
            # Example: Placeholder AU extraction
            aus = {
                'AU01': 0.0,  # Inner brow raiser
                'AU02': 0.0,  # Outer brow raiser
                'AU04': 0.0,  # Brow lowerer
                'AU05': 0.0,  # Upper lid raiser
                'AU06': 0.0,  # Cheek raiser
                'AU07': 0.0,  # Lid tightener
                'AU09': 0.0,  # Nose wrinkler
                'AU10': 0.0,  # Upper lip raiser
                'AU12': 0.0,  # Lip corner puller (smile)
                'AU14': 0.0,  # Dimpler
                'AU15': 0.0,  # Lip corner depressor
                'AU17': 0.0,  # Chin raiser
                'AU20': 0.0,  # Lip stretcher
                'AU23': 0.0,  # Lip tightener
                'AU25': 0.0,  # Lips part
                'AU26': 0.0,  # Jaw drop
                'AU45': 0.0   # Blink
            }

            return aus
        except Exception as e:
            print(f"Error in OpenFace AU extraction: {e}")
            return None

    def _extract_aus_fallback(self, frame):
        """
        Extract AUs using our fallback method (facial landmarks).

        Args:
            frame: Input frame

        Returns:
            aus: Dictionary of Action Units and their intensities
        """
        # This is a fallback method that approximates AUs based on
        # facial landmarks when OpenFace is not available

        # Detect face
        if isinstance(self.face_detector, dlib.fhog_object_detector):
            # Using dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector(gray)
            if not faces:
                return None

            # Get largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # Get facial landmarks
            if self.landmark_predictor:
                landmarks = self.landmark_predictor(gray, face)
                landmark_points = [(p.x, p.y) for p in landmarks.parts()]

                # Compute approximate AUs from landmarks
                aus = self._approximate_aus_from_landmarks(landmark_points, face)
                return aus

        else:
            # Using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return None

            # Get largest face
            face = max(faces, key=lambda rect: rect[2] * rect[3])

            # Since we don't have proper landmarks without dlib
            # return some basic features
            x, y, w, h = face
            aus = {
                'face_width': w,
                'face_height': h,
                'face_ratio': w/h,
                'face_area': w*h,
                'face_center_x': x + w/2,
                'face_center_y': y + h/2
            }

            return aus

    def _approximate_aus_from_landmarks(self, landmarks, face):
        """
        Approximate Action Units from facial landmarks.

        Args:
            landmarks: List of facial landmark coordinates
            face: Face bounding box

        Returns:
            aus: Dictionary of approximated Action Units
        """
        # This is a very simplified approximation of AUs based on
        # the relative positions of facial landmarks

        # Convert face rect to (x, y, w, h) format
        face_width = face.width()
        face_height = face.height()

        # Normalize landmarks by face size
        norm_landmarks = [(x/face_width, y/face_height) for x, y in landmarks]

        # We need at least 68 landmarks for the standard dlib facial landmarks
        if len(norm_landmarks) < 68:
            return None

        # Define landmark indices for different facial regions
        # (based on the standard 68-point facial landmark model)
        left_eye = norm_landmarks[36:42]
        right_eye = norm_landmarks[42:48]
        nose = norm_landmarks[27:36]
        mouth = norm_landmarks[48:68]
        jaw = norm_landmarks[0:17]
        left_eyebrow = norm_landmarks[17:22]
        right_eyebrow = norm_landmarks[22:27]

        # Compute approximate AUs
        aus = {}

        # AU01 + AU02: Inner + Outer Brow Raiser
        # Based on distance between eyebrows and eyes
        left_eyebrow_height = left_eyebrow[2][1] - left_eye[1][1]
        right_eyebrow_height = right_eyebrow[2][1] - right_eye[1][1]
        aus['AU01_AU02'] = -(left_eyebrow_height + right_eyebrow_height) / 2

        # AU04: Brow Lowerer
        # Based on distance between inner eyebrows
        inner_brow_distance = ((left_eyebrow[0][0] - right_eyebrow[4][0])**2 +
                              (left_eyebrow[0][1] - right_eyebrow[4][1])**2)**0.5
        aus['AU04'] = -inner_brow_distance

        # AU06 + AU07: Cheek Raiser + Lid Tightener
        # Based on eye opening
        left_eye_opening = ((left_eye[3][0] - left_eye[0][0])**2 +
                          (left_eye[3][1] - left_eye[0][1])**2)**0.5
        right_eye_opening = ((right_eye[3][0] - right_eye[0][0])**2 +
                           (right_eye[3][1] - right_eye[0][1])**2)**0.5
        aus['AU06_AU07'] = -(left_eye_opening + right_eye_opening) / 2

        # AU12: Lip Corner Puller (Smile)
        # Based on mouth width
        mouth_width = ((mouth[6][0] - mouth[0][0])**2 +
                     (mouth[6][1] - mouth[0][1])**2)**0.5
        aus['AU12'] = mouth_width

        # AU15: Lip Corner Depressor
        # Based on angle of mouth corners
        left_corner_angle = np.arctan2(mouth[0][1] - mouth[3][1],
                                    mouth[0][0] - mouth[3][0])
        right_corner_angle = np.arctan2(mouth[6][1] - mouth[3][1],
                                     mouth[6][0] - mouth[3][0])
        aus['AU15'] = -(left_corner_angle + right_corner_angle) / 2

        # AU25 + AU26: Lips Part + Jaw Drop
        # Based on mouth opening
        mouth_opening = ((mouth[3][0] - mouth[9][0])**2 +
                       (mouth[3][1] - mouth[9][1])**2)**0.5
        aus['AU25_AU26'] = mouth_opening

        # AU45: Blink
        # Based on eye aspect ratio
        left_eye_ratio = self._eye_aspect_ratio(left_eye)
        right_eye_ratio = self._eye_aspect_ratio(right_eye)
        aus['AU45'] = -(left_eye_ratio + right_eye_ratio) / 2

        return aus

    def _eye_aspect_ratio(self, eye):
        """
        Calculate the eye aspect ratio.

        Args:
            eye: List of eye landmark coordinates

        Returns:
            ratio: Eye aspect ratio
        """
        # Compute the euclidean distances between the vertical eye landmarks
        vert_dist1 = ((eye[1][0] - eye[5][0])**2 + (eye[1][1] - eye[5][1])**2)**0.5
        vert_dist2 = ((eye[2][0] - eye[4][0])**2 + (eye[2][1] - eye[4][1])**2)**0.5

        # Compute the euclidean distance between the horizontal eye landmarks
        horiz_dist = ((eye[0][0] - eye[3][0])**2 + (eye[0][1] - eye[3][1])**2)**0.5

        # Compute the eye aspect ratio
        ratio = (vert_dist1 + vert_dist2) / (2.0 * horiz_dist)

        return ratio

    def _extract_temporal_statistics(self, aus_sequence):
        """
        Extract temporal statistics from AU sequence.

        Args:
            aus_sequence: Sequence of AUs across frames

        Returns:
            features: Temporal features
        """
        # Convert dictionary-based AUs to np array if needed
        if isinstance(aus_sequence[0], dict):
            # Get all keys
            keys = sorted(aus_sequence[0].keys())
            # Convert to array
            aus_array = np.zeros((len(aus_sequence), len(keys)))
            for i, aus in enumerate(aus_sequence):
                for j, key in enumerate(keys):
                    aus_array[i, j] = aus.get(key, 0.0)

            aus_sequence = aus_array

        # Calculate temporal statistics
        # Mean, std, min, max, range, etc.
        mean_aus = np.mean(aus_sequence, axis=0)
        std_aus = np.std(aus_sequence, axis=0)
        min_aus = np.min(aus_sequence, axis=0)
        max_aus = np.max(aus_sequence, axis=0)
        range_aus = max_aus - min_aus

        # First-order derivatives (changes between frames)
        if len(aus_sequence) > 1:
            derivatives = np.diff(aus_sequence, axis=0)
            mean_derivatives = np.mean(derivatives, axis=0)
            std_derivatives = np.std(derivatives, axis=0)
            max_derivatives = np.max(np.abs(derivatives), axis=0)
        else:
            mean_derivatives = np.zeros_like(mean_aus)
            std_derivatives = np.zeros_like(std_aus)
            max_derivatives = np.zeros_like(max_aus)

        # Combine all statistics into a single feature vector
        features = np.concatenate([
            mean_aus, std_aus, min_aus, max_aus, range_aus,
            mean_derivatives, std_derivatives, max_derivatives
        ])

        return features

    def _reduce_dimensionality(self, aus_sequence):
        """
        Reduce dimensionality of AU sequence.

        Args:
            aus_sequence: Sequence of AUs across frames

        Returns:
            features: Reduced-dimension features
        """
        # Extract temporal statistics first
        stats_features = self._extract_temporal_statistics(aus_sequence)

        # If the dimension is already small enough return as is
        if len(stats_features) <= self.feature_dim:
            return stats_features

        # Otherwise apply PCA or other dimensionality reduction
        # This is a placeholder - we'd use sklearn.decomposition.PCA in practice
        # For now just return a subset of features
        features = stats_features[:self.feature_dim]

        return features

    def _visualize_aus(self, frame, aus, video_name, frame_num, output_dir):
        """
        Visualize the extracted AUs on the frame.

        Args:
            frame: Input frame
            aus: Dictionary of Action Units
            video_name: Name of the source video
            frame_num: Frame number
            output_dir: Directory to save visualization
        """
        if aus is None:
            return

        # Create output directory if it doesn't exist
        viz_dir = os.path.join(output_dir, 'au_visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Create a copy of the frame for visualization
        viz_frame = frame.copy()

        # Add AU values as text
        y_offset = 30
        for i, (au, value) in enumerate(sorted(aus.items())):
            text = f"{au}: {value:.2f}"
            cv2.putText(viz_frame, text, (10, y_offset + i*30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save the visualization
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        output_file = os.path.join(viz_dir, f"{base_name}_frame{frame_num:04d}.jpg")
        cv2.imwrite(output_file, cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR))

    def batch_extract_features(self, video_paths, output_dir, batch_size=32,
                              visualize=False, resume=False):
        """
        Extract features from multiple videos and save to disk.

        Args:
            video_paths: List of video file paths
            output_dir: Directory to save extracted features
            batch_size: Batch size for processing
            visualize: Whether to save visualizations
            resume: Whether to skip already processed files

        Returns:
            results: Dictionary mapping video paths to success/failure
        """
        features_dir = os.path.join(output_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)

        if visualize:
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)

        # Track successes, failures, and skips
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        failures = []

        # Process videos
        for i, video_path in enumerate(tqdm(video_paths, desc="Extracting FACS features")):
            try:
                # Create output filename
                base_name = os.path.basename(video_path)
                base_name = os.path.splitext(base_name)[0]
                output_path = os.path.join(features_dir, f"{base_name}.npy")

                # Skip if file exists and resume is True
                if resume and os.path.exists(output_path):
                    results['skipped'] += 1
                    continue

                # Extract features
                features = self.extract_features(
                    video_path,
                    save_visualization=(visualize and i % 50 == 0),
                    output_dir=viz_dir if visualize else None
                )

                if features is not None:
                    # Save features
                    np.save(output_path, features)
                    results['success'] += 1

                    # Print progress periodically
                    if (i + 1) % 100 == 0:
                        print(f"Progress: {i+1}/{len(video_paths)} - "
                             f"Success: {results['success']} "
                             f"Failed: {results['failed']} "
                             f"Skipped: {results['skipped']}")
                else:
                    results['failed'] += 1
                    failures.append(video_path)

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                results['failed'] += 1
                failures.append(f"{video_path}: {str(e)}")

        # Save extraction summary
        summary_path = os.path.join(output_dir, "facs_extraction_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"FACS Feature Extraction Summary:\n")
            f.write(f"------------------------------\n")
            f.write(f"Total files: {len(video_paths)}\n")
            f.write(f"Successful extractions: {results['success']}\n")
            f.write(f"Failed extractions: {results['failed']}\n")
            f.write(f"Skipped files: {results['skipped']}\n\n")

            if failures:
                f.write("Failed Files:\n")
                f.write("------------\n")
                for failure in failures:
                    f.write(f"{failure}\n")

        print(f"\nFACS extraction complete!")
        print(f"Total: {len(video_paths)} Success: {results['success']} "
             f"Failed: {results['failed']} Skipped: {results['skipped']}")
        print(f"Summary saved to {summary_path}")

        return results


def test_feature_extractor():
    """Test the FACS feature extractor."""
    # Create a feature extractor
    extractor = FACSFeatureExtractor()

    # Sample video path
    video_path = "./downsampled_videos/RAVDESS/Actor_01/01-01-06-01-02-01-01.mp4"

    if os.path.exists(video_path):
        # Extract features
        features = extractor.extract_features(video_path, save_visualization=True,
                                           output_dir="./processed_data/facs_visualizations")

        if features is not None:
            print(f"Extracted FACS features shape: {features.shape}")

            # Save features
            os.makedirs("./processed_data/facs_features", exist_ok=True)
            np.save("./processed_data/facs_features/sample.npy", features)
    else:
        print(f"Sample video not found: {video_path}")

        # Try to find any video file
        import glob
        video_files = glob.glob("./downsampled_videos/RAVDESS/**/*.mp4", recursive=True)
        if video_files:
            print(f"Found {len(video_files)} video files. Testing with {video_files[0]}")
            features = extractor.extract_features(video_files[0], save_visualization=True,
                                               output_dir="./processed_data/facs_visualizations")

            if features is not None:
                print(f"Extracted FACS features shape: {features.shape}")


if __name__ == "__main__":
    test_feature_extractor()
