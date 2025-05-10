#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_audio_manifest.py

Scans the RAVDESS and CREMA‑D video directories, extracts emotion
labels from each filename, and writes a tab‑separated manifest:

    path<TAB>label_id

It also outputs class_counts.json so that the training script can
compute class‑weighted loss.

Usage
-----
python scripts/build_audio_manifest.py \
        --ravdess_dir data/RAVDESS \
        --cremad_dir data/CREMA-D \
        --out_tsv  data/audio_manifest.tsv \
        --out_json data/class_counts.json

Options
-------
--extract-wav
    If supplied, a 16 kHz mono .wav is extracted next to each video
    and the manifest will point to the .wav paths; otherwise it stores
    the original video paths (later streamed via ffmpeg).

Dependencies
------------
ffmpeg (command‑line) and optionally moviepy for wav extraction.
"""

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# ---------- label parsing helpers -------------------------------------------------

EMOTION_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
RAVDESS_EMOTION_MAP = {
    "01": "NEU",
    "02": "NEU",  # calm→neutral bucket
    "03": "HAP",
    "04": "SAD",
    "05": "ANG",
    "06": "FEA",
    "07": "DIS",
    "08": "HAP",  # surprised→use happy bucket (rare)
}


def parse_ravdess_label(file_stem: str) -> int:
    """
    RAVDESS filename pattern:
        XX-XX-EMO-...  3rd group = two‑digit code
    """
    parts = file_stem.split("-")
    if len(parts) < 3:
        raise ValueError("Unexpected RAVDESS filename: %s" % file_stem)
    code = parts[2]
    emo = RAVDESS_EMOTION_MAP.get(code)
    return EMOTION_MAP[emo]


def parse_cremad_label(file_stem: str) -> int:
    """
    CREMA‑D filename pattern:
        ID_EMO_INT.wav  2nd group = three‑char emotion code
    """
    parts = file_stem.split("_")
    if len(parts) < 3:
        raise ValueError("Unexpected CREMA‑D filename: %s" % file_stem)
    emo = parts[2]
    return EMOTION_MAP[emo]


# ---------- wav extraction --------------------------------------------------------


def extract_wav(video_path: Path, wav_path: Path) -> None:
    """Extract 16 kHz mono wav using ffmpeg"""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        "-loglevel",
        "error",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


# ---------- main ------------------------------------------------------------------


def collect_files(
    ravdess_dir: Path, cremad_dir: Path
) -> List[Tuple[Path, int, bool]]:
    """
    Returns list of (path, label_id, is_ravdess)
    """
    files = []
    # RAVDESS
    for video in ravdess_dir.glob("Actor_*/*.mp4"):
        label_id = parse_ravdess_label(video.stem)
        files.append((video, label_id, True))
    # CREMA‑D
    for video in cremad_dir.glob("*.flv"):
        label_id = parse_cremad_label(video.stem)
        files.append((video, label_id, False))
    return files


def main():
    parser = argparse.ArgumentParser(description="Build audio manifest TSV")
    parser.add_argument("--ravdess_dir", required=True, type=Path)
    parser.add_argument("--cremad_dir", required=True, type=Path)
    parser.add_argument("--out_tsv", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument(
        "--extract-wav",
        action="store_true",
        help="Extract 16 kHz wav and reference those in manifest",
    )
    args = parser.parse_args()

    entries = collect_files(args.ravdess_dir, args.cremad_dir)
    if not entries:
        print("No video files found – check paths.")
        return

    class_counts = Counter()
    with args.out_tsv.open("w", encoding="utf-8") as tsv:
        for path, label, is_ravdess in entries:
            class_counts[label] += 1
            if args.extract_wav:
                wav_name = path.with_suffix(".wav").name
                wav_out = path.parent / wav_name
                if not wav_out.exists():
                    try:
                        extract_wav(path, wav_out)
                    except subprocess.CalledProcessError as e:
                        print(f"ffmpeg failed on {path}: {e}")
                        continue
                manifest_path = wav_out
            else:
                manifest_path = path
            tsv.write(f"{manifest_path}\t{label}\n")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as json_f:
        json.dump(class_counts, json_f, indent=2)
    print(f"Wrote {args.out_tsv} with {len(entries)} entries.")
    print(f"Class counts: {dict(class_counts)}")


if __name__ == "__main__":
    main()
