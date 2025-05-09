#!/usr/bin/env python3
"""
Test script for the FaceNet extractor.
This script allows testing the FaceNet extractor on sample images or videos.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_extractor import FaceNetExtractor

def test_on_video(video_path, show_frames=False):
    """
    Test the FaceNetExtractor on a video file.
    
    Args:
        video_path: Path to the video file
        show_frames: Whether to show frames with detected faces
    """
    print(f"\nTesting FaceNetExtractor on video: {os.path.basename(video_path)}")
    
    # Initialize extractor
    extractor = FaceNetExtractor(keep_all=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_count} frames at {fps} fps")
    
    # Process video
    frame_idx = 0
    detection_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame (for speed)
        if frame_idx % 5 == 0:
            # First convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes, probs = extractor.mtcnn.detect(rgb_frame)
            
            # Check if faces were detected
            face_detected = boxes is not None and len(boxes) > 0
            detection_results.append(face_detected)
            
            if show_frames and face_detected:
                # Draw bounding boxes
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Frame', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        
        frame_idx += 1
        
        # Print progress every 20 frames
        if frame_idx % 20 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Compute statistics
    total_processed = len(detection_results)
    faces_detected = sum(detection_results)
    detection_rate = faces_detected / total_processed * 100 if total_processed > 0 else 0
    
    print(f"\nResults:")
    print(f"Processed {total_processed} frames")
    print(f"Frames with detected faces: {faces_detected} ({detection_rate:.1f}%)")
    
    return detection_results

def test_on_image(image_path, show_result=False):
    """
    Test the FaceNetExtractor on a single image.
    
    Args:
        image_path: Path to the image file
        show_result: Whether to show the image with detected faces
    """
    print(f"\nTesting FaceNetExtractor on image: {os.path.basename(image_path)}")
    
    # Initialize extractor
    extractor = FaceNetExtractor(keep_all=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes, probs = extractor.mtcnn.detect(rgb_image)
    
    # Check if faces were detected
    if boxes is not None and len(boxes) > 0:
        print(f"Detected {len(boxes)} faces")
        
        # Extract features if at least one face is detected
        embedding = extractor.extract_features(image)
        
        print(f"Embedding shape: {embedding.shape}")
        print(f"Non-zero elements: {np.count_nonzero(embedding)}/{embedding.size} "
              f"({np.count_nonzero(embedding)/embedding.size*100:.1f}%)")
        
        if show_result:
            # Draw bounding boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if probs is not None:
                    cv2.putText(image, f"{probs[i]:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show image
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Detected {len(boxes)} faces")
            plt.axis('off')
            plt.show()
    else:
        print("No faces detected")
        
        # Try extracting features anyway to test zero-handling
        embedding = extractor.extract_features(image)
        print(f"Embedding shape: {embedding.shape}")
        print(f"All zeros: {np.all(embedding == 0)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FaceNetExtractor on images or videos")
    parser.add_argument("--image", help="Path to an image file for testing")
    parser.add_argument("--video", help="Path to a video file for testing")
    parser.add_argument("--show", action="store_true", help="Show frames/images with detected faces")
    parser.add_argument("--ravdess-sample", action="store_true", help="Test on a sample RAVDESS video")
    parser.add_argument("--crema-d-sample", action="store_true", help="Test on a sample CREMA-D video")
    
    args = parser.parse_args()
    
    if args.image:
        test_on_image(args.image, args.show)
    elif args.video:
        test_on_video(args.video, args.show)
    elif args.ravdess_sample:
        # Find a sample RAVDESS video
        import glob
        ravdess_videos = glob.glob("downsampled_videos/RAVDESS/*/*.mp4")
        if ravdess_videos:
            test_on_video(ravdess_videos[0], args.show)
        else:
            print("No RAVDESS videos found in 'downsampled_videos/RAVDESS/'")
    elif args.crema_d_sample:
        # Find a sample CREMA-D video
        import glob
        crema_videos = glob.glob("downsampled_videos/CREMA-D-audio-complete/*.flv")
        if crema_videos:
            test_on_video(crema_videos[0], args.show)
        else:
            print("No CREMA-D videos found in 'downsampled_videos/CREMA-D-audio-complete/'")
    else:
        print("Please provide either --image, --video, --ravdess-sample, or --crema-d-sample argument")

if __name__ == "__main__":
    main()
