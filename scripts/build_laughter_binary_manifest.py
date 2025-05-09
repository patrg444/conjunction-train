import os
import csv

# Paths
AUDIO_DIR = "datasets/vocalsounds/audio_16k"
LAUGHTER_MANIFEST = "datasets/vocalsounds/laughter_manifest.csv"
OUTPUT_MANIFEST = "datasets/vocalsounds/laughter_binary_manifest.csv"

def get_all_wav_files(audio_dir):
    wav_files = []
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    return set(os.path.normpath(p) for p in wav_files)

def get_laughter_files(manifest_path):
    laughter_files = set()
    with open(manifest_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            abs_path = line.split(",")[0]
            # Convert to relative path from project root
            if abs_path.startswith("/Users/"):
                rel_path = os.path.relpath(abs_path, os.getcwd())
            else:
                rel_path = abs_path
            laughter_files.add(os.path.normpath(rel_path))
    return laughter_files

def main():
    all_wavs = get_all_wav_files(AUDIO_DIR)
    laughter_wavs = get_laughter_files(LAUGHTER_MANIFEST)

    # Positive: all in laughter manifest
    positives = []
    with open(LAUGHTER_MANIFEST, "r") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            abs_path, start, end, label = row
            if abs_path.startswith("/Users/"):
                rel_path = os.path.relpath(abs_path, os.getcwd())
            else:
                rel_path = abs_path
            rel_path = os.path.normpath(rel_path)
            positives.append([rel_path, start, end, "1"])

    # Negative: all .wav not in laughter manifest
    negatives = []
    for wav in sorted(all_wavs - laughter_wavs):
        negatives.append([wav, "0.0", "-1", "0"])

    print(f"Found {len(positives)} laughter files, {len(negatives)} non-laughter files.")

    # Write combined manifest
    with open(OUTPUT_MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        for row in positives:
            writer.writerow(row)
        for row in negatives:
            writer.writerow(row)

    print(f"Wrote {len(positives) + len(negatives)} entries to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
