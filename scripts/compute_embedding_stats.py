#!/usr/bin/env python3
"""
Compute FaceNet embedding normalization stats from live camera frames.
Saves embedding_mean.npy and embedding_std.npy for use in streaming_demo_audio_pooling_lstm.py.
"""

import argparse
import numpy as np
import cv2
from facenet_extractor import FaceNetExtractor

def main(count, device):
    extractor = FaceNetExtractor()
    embeddings = []

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Failed to open camera index {device}")
        return

    print(f"Capturing {count} frames for embedding stats...")
    while len(embeddings) < count:
        ret, frame = cap.read()
        if not ret:
            continue
        emb = extractor.extract_features(frame)
        embeddings.append(emb)

    cap.release()

    arr = np.stack(embeddings, axis=0)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0) + 1e-8

    np.save("embedding_mean.npy", mean)
    np.save("embedding_std.npy", std)
    print("Saved embedding_mean.npy and embedding_std.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FaceNet embedding normalization stats")
    parser.add_argument("--count", type=int, default=200,
                        help="Number of frames to sample")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index for cv2.VideoCapture")
    args = parser.parse_args()
    main(args.count, args.device)
