#!/usr/bin/env python3
"""
extract_inference_features.py

Combined feature-extraction utility for the RAVDESS/CREMA-D fusion model.

Usage
-----
python extract_inference_features.py \\
       --audio path/to/audio.wav \\
       --video path/to/video.mp4 \\
       --out_file sample_features.npz

If only an audio or only a video file is supplied, the missing branch will be
filled with an empty (zero-row) array so the .npz structure is always valid.

The script re-uses:
  • FaceNetExtractor  – video branch (512-dim embeddings / frame)
  • HuBERT large       – audio branch (1024-dim embeddings / frame)

Output .npz structure  (matches training files)
------------------------------------------------
audio_features        : float32  [T_audio, 1024]
video_features        : float32  [T_video, 512]
audio_timestamps      : float32  [T_audio]
video_timestamps      : float32  [T_video]
params                : dict     (meta info)
"""

import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, List

# ---------------------------------------------------------------------------
# Video branch  (FaceNet)
# ---------------------------------------------------------------------------
try:
    from scripts.extraction.feature_extraction.facenet_extractor import FaceNetExtractor
except ModuleNotFoundError:
    # Fallback if running inside package
    from fusion_emotion_model_marketplace_package.src.scripts.extraction.feature_extraction.facenet_extractor import FaceNetExtractor

# ---------------------------------------------------------------------------
# Audio branch  (HuBERT)
# ---------------------------------------------------------------------------
from transformers import HubertModel, Wav2Vec2FeatureExtractor


def load_audio(
    file_path: Path,
    target_sr: int = 16000,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return (waveform[T], timestamps[T])"""
    waveform, sr = torchaudio.load(str(file_path))
    if waveform.shape[0] > 1:  # mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # squeeze channel, move to device
    wav = waveform.squeeze(0).to(device)
    duration = wav.shape[0] / target_sr
    timestamps = np.linspace(0, duration, wav.shape[0], dtype=np.float32)
    return wav, timestamps


def extract_audio_features(
    wav: torch.Tensor,
    model: HubertModel,
    processor: Wav2Vec2FeatureExtractor,
    device: torch.device,
) -> torch.Tensor:
    """Return tensor [T_frames, hidden_size]"""
    inputs = processor(wav.cpu().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
    return outputs.last_hidden_state[0].cpu()  # remove batch dim


# ---------------------------------------------------------------------------
# Video helper
# ---------------------------------------------------------------------------
def extract_video_features(video_path: Path, extractor: FaceNetExtractor, sample_interval: int = 1):
    feats, ts = extractor.process_video(str(video_path), sample_interval=sample_interval)
    if feats is None:
        return np.zeros((0, extractor.embedding_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return feats.astype(np.float32), ts.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract audio+video features into .npz for fusion model.")
    parser.add_argument("--audio", type=str, help="Path to .wav file", required=False)
    parser.add_argument("--video", type=str, help="Path to video (.mp4/.avi)", required=False)
    parser.add_argument("--out_file", type=str, required=True, help="Destination .npz path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_interval", type=int, default=1, help="Process every Nth video frame")
    parser.add_argument("--hubert_ckpt", default="facebook/hubert-large-ls960-ft")
    args = parser.parse_args()

    device = torch.device(args.device)

    # -------------------- audio branch --------------------
    if args.audio:
        wav, audio_ts = load_audio(Path(args.audio), device=device)
        proc = Wav2Vec2FeatureExtractor.from_pretrained(args.hubert_ckpt)
        hubert = HubertModel.from_pretrained(args.hubert_ckpt).to(device).eval()
        audio_emb = extract_audio_features(wav, hubert, proc, device)
        audio_arr = audio_emb.numpy().astype(np.float32)
    else:
        audio_arr = np.zeros((0, 1024), dtype=np.float32)
        audio_ts = np.zeros((0,), dtype=np.float32)

    # -------------------- video branch --------------------
    if args.video:
        extractor = FaceNetExtractor(device=device)
        video_arr, video_ts = extract_video_features(Path(args.video), extractor, args.sample_interval)
    else:
        video_arr = np.zeros((0, 512), dtype=np.float32)
        video_ts = np.zeros((0,), dtype=np.float32)

    # -------------------- save ---------------------------
    np.savez(
        args.out_file,
        audio_features=audio_arr,
        video_features=video_arr,
        audio_timestamps=audio_ts,
        video_timestamps=video_ts,
        params=dict(
            audio_ckpt=args.hubert_ckpt,
            video_model="FaceNet (facenet-pytorch, vggface2)",
            sample_interval=args.sample_interval,
        ),
    )
    print(f"Saved features to {args.out_file}")


if __name__ == "__main__":
    main()
