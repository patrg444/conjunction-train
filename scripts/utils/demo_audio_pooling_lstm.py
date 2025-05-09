#!/usr/bin/env python3
"""
Demo for Audio Pooling LSTM model.
Supports static file inference and live streaming.
Handles models with either one combined input or two separate inputs.
"""

import os
import argparse
import subprocess
import time
import numpy as np
from tensorflow.keras.models import load_model

from facenet_extractor import FaceNetExtractor
from multimodal_preprocess_fixed import extract_audio_functionals

# Default model path
MODEL_PATH = "target_model/model_85.6_accuracy.keras"
EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]

def extract_video_features(video_path):
    extractor = FaceNetExtractor(keep_all=False)
    feats, _ = extractor.process_video(video_path)
    if feats is None or len(feats)==0:
        raise RuntimeError("No video features extracted.")
    return feats.astype(np.float32)

def extract_audio_features(audio_path):
    feats, _ = extract_audio_functionals(audio_path)
    if feats is None or len(feats)==0:
        raise RuntimeError("No audio features extracted.")
    return feats.astype(np.float32)

def pool_audio_to_video(audio_feats, num_vid_frames):
    if audio_feats.shape[0]==0 or num_vid_frames==0:
        return np.zeros((num_vid_frames, audio_feats.shape[1]), dtype=np.float32)
    pooled = []
    total = audio_feats.shape[0]
    for i in range(num_vid_frames):
        start = int(i * total / num_vid_frames)
        end   = min(int((i+1)*total/num_vid_frames), total)
        seg = audio_feats[start:end] if end>start else audio_feats[start:start+1]
        pooled.append(np.mean(seg, axis=0))
    return np.stack(pooled, axis=0)

def normalize(feats, mean, std):
    std = np.where(std==0, 1.0, std)
    return (feats - mean) / std

def run_inference(model, vid_path, aud_path):
    print("Running inference pipeline...")
    # resample video
    tmp_vid = "video_resampled.mp4"
    subprocess.run(["ffmpeg","-y","-i",vid_path,"-r","15",tmp_vid],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Video resampled to 15 fps:", tmp_vid)
    vid_feats = extract_video_features(tmp_vid)
    print(f"Extracted {vid_feats.shape[0]} video frames")
    os.remove(tmp_vid)
    aud_feats = extract_audio_features(aud_path)
    print(f"Extracted {aud_feats.shape[0]} audio frames")

    # load stats
    audio_mean = np.load("audio_mean.npy")
    audio_std  = np.load("audio_std.npy")
    video_mean = np.load("video_mean.npy")
    video_std  = np.load("video_std.npy")

    vid_norm = normalize(vid_feats, video_mean, video_std)
    aud_norm = normalize(aud_feats, audio_mean, audio_std)
    pooled   = pool_audio_to_video(aud_norm, vid_norm.shape[0])

    # prepare model inputs
    if len(model.inputs)==2:
        # separate streams
        Xvid = np.expand_dims(vid_norm, axis=0)
        Xaud = np.expand_dims(aud_norm, axis=0)
        inputs = [Xvid, Xaud]
    else:
        # combined
        comb = np.expand_dims(np.concatenate([vid_norm, pooled], axis=1), axis=0)
        inputs = comb

    print("Running model.predict...")
    pred = model.predict(inputs, verbose=0)[0]
    idx  = np.argmax(pred)
    print(f"Predicted emotion: {EMOTIONS[idx]}, Confidence: {100*pred[idx]:.2f}%")
    print("Inference done")

def live_stream(model, duration, step):
    print(f"Live demo: window={duration}s, step={step}s")
    while True:
        start = time.time()
        print(f"> Capturing video and audio for {duration}s...")
        # record video+audio with progress dots
        capture_cmd = [
            "ffmpeg","-y","-t",str(duration),
            "-f","avfoundation","-framerate","30","-pixel_format","uyvy422","-vsync","2",
            "-video_size","640x480","-i","0:1","live_cam.mp4"
        ]
        print("> ffmpeg capture cmd:", " ".join(capture_cmd))
        proc = subprocess.Popen(capture_cmd)
        # show progress dots until capture completes
        while proc.poll() is None:
            print(".", end="", flush=True)
            time.sleep(0.5)
        print(" done")
        # extract audio
        extract_cmd = [
            "ffmpeg","-y","-i","live_cam.mp4","-vn",
            "-acodec","pcm_s16le","live_cam.wav"
        ]
        print("> ffmpeg extract audio cmd:", " ".join(extract_cmd))
        subprocess.run(extract_cmd, check=True)
        # debug capture
        print("Debug: live_cam.mp4 exists:", os.path.exists("live_cam.mp4"),
              "size:", os.path.getsize("live_cam.mp4") if os.path.exists("live_cam.mp4") else 0)
        print("Debug: live_cam.wav exists:", os.path.exists("live_cam.wav"),
              "size:", os.path.getsize("live_cam.wav") if os.path.exists("live_cam.wav") else 0)
        try:
            run_inference(model, "live_cam.mp4", "live_cam.wav")
        except Exception as e:
            print("Error:", e)
        # cleanup
        for f in ("live_cam.mp4","live_cam.wav"):
            if os.path.exists(f): os.remove(f)
        delta = time.time()-start
        if delta < step: time.sleep(step-delta)

def main():
    p = argparse.ArgumentParser(description="Audio Pooling LSTM Demo")
    p.add_argument("--video", type=str, help="Video file path")
    p.add_argument("--audio", type=str, help="Audio file path")
    p.add_argument("--model", type=str, default=MODEL_PATH, help="Model file")
    p.add_argument("--live", action="store_true", help="Live demo")
    p.add_argument("--duration", type=int, default=2, help="Window sec")
    p.add_argument("--step", type=int, default=1, help="Interval sec")
    args = p.parse_args()

    model = load_model(args.model, compile=False)

    if args.live:
        live_stream(model, args.duration, args.step)
    else:
        if not args.video or not args.audio:
            p.error("--video and --audio required")
        run_inference(model, args.video, args.audio)

if __name__=="__main__":
    main()
