#!/usr/bin/env python3
"""
Streaming demo for Audio Pooling LSTM model with overlapping windows.
Captures raw video frames and audio samples from avfoundation via ffmpeg pipes,
maintains ring buffers, extracts FaceNet embeddings per frame, normalizes them,
and runs inference on overlapping windows.
"""

import argparse
import subprocess
import threading
import time
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

from facenet_extractor import FaceNetExtractor

# Settings
VIDEO_DEVICE = "0:1"
WIDTH, HEIGHT, FPS = 640, 480, 30
SR = 48000
WINDOW_SEC = 2
STEP_SEC = 1

# Model and stats
MODEL_PATH = "target_model/model_82.9_accuracy.h5"
EMOTIONS = ["anger","disgust","fear","happiness","neutral","sadness"]
audio_mean = np.load("audio_mean.npy")
audio_std  = np.load("audio_std.npy")

# FaceNet extractor
extractor = FaceNetExtractor()
EMBED_DIM = extractor.embedding_dim
embedding_mean = np.load("embedding_mean.npy")
embedding_std  = np.load("embedding_std.npy")

def normalize(arr, mean, std):
    std_safe = np.where(std == 0, 1.0, std)
    return (arr - mean) / std_safe

def inference_on_buffer(model, vid_buf, aud_buf):
    # Extract embeddings per frame
    frames = list(vid_buf)
    embeddings = np.stack(
        [extractor.extract_features(frame) for frame in frames],
        axis=0
    ).astype(np.float32)  # shape (T, EMBED_DIM)
    # normalize embeddings by precomputed stats
    vid_norm = normalize(embeddings, embedding_mean, embedding_std)

    # Audio array
    aud_arr = np.array(list(aud_buf), dtype=np.float32).reshape(-1,1)
    aud_norm = normalize(aud_arr, audio_mean, audio_std)

    # Pool audio to video frames
    step = aud_norm.shape[0] / vid_norm.shape[0]
    pooled = []
    for i in range(vid_norm.shape[0]):
        start = int(i * step)
        end = int((i+1) * step)
        pooled.append(np.mean(aud_norm[start:end], axis=0))
    pooled = np.stack(pooled, axis=0)

    # Prepare inputs
    if len(model.inputs) == 2:
        Xvid = np.expand_dims(vid_norm, 0)
        Xaud = np.expand_dims(aud_norm, 0)
        inputs = [Xvid, Xaud]
    else:
        comb = np.expand_dims(np.concatenate([vid_norm, pooled], axis=1), 0)
        inputs = comb

    pred = model.predict(inputs, verbose=0)[0]
    idx  = np.argmax(pred)
    ts   = time.strftime('%H:%M:%S')
    scores = ', '.join([f"{emo}:{prob*100:.1f}%" for emo, prob in zip(EMOTIONS, pred)])
    print(f"[{ts}] Confidences: {scores}")
    print(f"[{ts}] Predicted: {EMOTIONS[idx]} ({pred[idx]*100:.1f}%)")

def capture_loop(vid_buf, aud_buf, stop_event):
    vid_dev, aud_dev = VIDEO_DEVICE.split(":")
    # Video pipe
    vc = [
        "ffmpeg","-f","avfoundation","-framerate",str(FPS),
        "-pixel_format","uyvy422","-vsync","2",
        "-video_size",f"{WIDTH}x{HEIGHT}","-i",f"{vid_dev}:none",
        "-f","rawvideo","-pix_fmt","rgb24",
        "-hide_banner","-nostats","-loglevel","error","-"
    ]
    # Audio pipe
    ac = [
        "ffmpeg","-hide_banner","-nostats","-loglevel","error",
        "-f","avfoundation","-i",f"none:{aud_dev}",
        "-ar",str(SR),"-ac","1",
        "-f","f32le","-acodec","pcm_f32le","-"
    ]
    vproc = subprocess.Popen(vc, stdout=subprocess.PIPE)
    aproc = subprocess.Popen(ac, stdout=subprocess.PIPE)

    frame_bytes = WIDTH * HEIGHT * 3
    sample_bytes = 4  # float32
    chunk_samples = SR // FPS  # per-frame audio samples

    try:
        while not stop_event.is_set():
            # Read one video frame
            data = vproc.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                break
            frame = np.frombuffer(data, np.uint8).reshape(HEIGHT, WIDTH, 3)
            vid_buf.append(frame)

            # Read audio chunk
            aud_data = aproc.stdout.read(chunk_samples * sample_bytes)
            if len(aud_data) == 0:
                break
            samples = np.frombuffer(aud_data, np.float32)
            for s in samples:
                aud_buf.append([s])
    finally:
        vproc.terminate()
        aproc.terminate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=WINDOW_SEC)
    parser.add_argument("--step",   type=int, default=STEP_SEC)
    parser.add_argument("--model",  type=str, default=MODEL_PATH)
    args = parser.parse_args()

    model = load_model(args.model, compile=False)
    vid_buf = deque(maxlen=args.window * FPS)
    aud_buf = deque(maxlen=args.window * SR)
    stop_event = threading.Event()

    t = threading.Thread(target=capture_loop, args=(vid_buf, aud_buf, stop_event))
    t.daemon = True
    t.start()

    print(f"Waiting to fill initial buffer: target {args.window*FPS} video frames and {args.window*SR} audio samples")

    try:
        while True:
            time.sleep(args.step)
            if len(vid_buf) == args.window * FPS and len(aud_buf) >= args.window * SR:
                print(f"[{time.strftime('%H:%M:%S')}] Triggering inference on sliding window")
                start_inf = time.time()
                inference_on_buffer(model, vid_buf, aud_buf)
                inf_dur = (time.time() - start_inf) * 1000
                print(f"[{time.strftime('%H:%M:%S')}] Inference runtime: {inf_dur:.1f} ms")
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping streaming demo.")

if __name__ == "__main__":
    main()
