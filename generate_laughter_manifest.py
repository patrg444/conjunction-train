#!/usr/bin/env python3
"""
Generate a realistic laughter manifest file for emotion recognition training.

This script creates a CSV file that maps audio/video filenames to laughter labels (0 or 1).
Each row represents one sample with its corresponding laughter label.

The manifest follows the expected format for the `train_audio_pooling_lstm_with_laughter.py` script.
"""

import os
import csv
import random
import argparse
from pathlib import Path

def generate_manifest(output_file, num_samples=500, laughter_ratio=0.25):
    """
    Generate a laughter manifest CSV file.

    Args:
        output_file: Path to the output CSV file
        num_samples: Number of samples to generate
        laughter_ratio: Ratio of laughter samples (0-1)
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate sample filenames and labels
    samples = []

    # Define realistic sample sources and their base filenames
    sources = [
        # RAVDESS samples (Actor_01 to Actor_24)
        {"prefix": "Actor_", "range": range(1, 25), "formats": [
            "{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}.wav",  # Format: 01-01-01-01-01-01-01.wav
        ]},
        # CREMA-D samples
        {"prefix": "", "range": range(1, 100), "formats": [
            "CremaD_{:04d}_IEO.wav",  # Format: CremaD_0001_IEO.wav
            "CremaD_{:04d}_TIE.wav",  # Format: CremaD_0002_TIE.wav
            "CremaD_{:04d}_MTI.wav",  # Format: CremaD_0003_MTI.wav
        ]},
        # AudioSet laughter samples
        {"prefix": "audioset_laugh_", "range": range(1, 100), "formats": [
            "{:04d}.wav",  # Format: audioset_laugh_0001.wav
        ]},
    ]

    # Generate samples
    for i in range(num_samples):
        # Randomly select a source
        source = random.choice(sources)

        # Generate a filename
        actor_id = random.choice(list(source["range"]))
        format_template = random.choice(source["formats"])

        if "Actor_" in source["prefix"]:
            # For RAVDESS: emotion_id, intensity, statement, repetition, gender
            emotion_id = random.randint(1, 8)
            intensity = random.randint(1, 2)
            statement = random.randint(1, 2)
            repetition = random.randint(1, 2)
            gender = 1 if actor_id <= 12 else 2  # Actors 1-12 are male, 13-24 are female

            filename = source["prefix"] + f"{actor_id:02d}/" + format_template.format(
                emotion_id, intensity, statement, repetition, actor_id, gender, repetition
            )
        else:
            # For other sources
            filename = source["prefix"] + format_template.format(actor_id)

        # Determine if this sample has laughter
        # Make laughter more likely in certain sources
        has_laughter = 0
        if "audioset_laugh_" in source["prefix"]:
            # AudioSet laugh samples have a higher probability of laughter
            has_laughter = 1 if random.random() < 0.8 else 0
        elif "CremaD" in format_template:
            # CREMA-D samples have a medium probability of laughter
            has_laughter = 1 if random.random() < 0.2 else 0
        else:
            # RAVDESS samples have a lower probability of laughter
            has_laughter = 1 if random.random() < 0.1 else 0

        # Ensure we get the desired overall ratio
        if i < int(num_samples * laughter_ratio):
            has_laughter = 1

        # Add split information (train, val, test)
        if i < int(num_samples * 0.7):
            split = "train"
        elif i < int(num_samples * 0.85):
            split = "val"
        else:
            split = "test"

        samples.append({"filepath": filename, "laugh": has_laughter, "split": split})

    # Randomize the order
    random.shuffle(samples)

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "laugh", "split"])
        writer.writeheader()
        writer.writerows(samples)

    print(f"Generated {num_samples} samples in {output_file}")

    # Count statistics
    train_count = sum(1 for s in samples if s["split"] == "train")
    val_count = sum(1 for s in samples if s["split"] == "val")
    test_count = sum(1 for s in samples if s["split"] == "test")

    laugh_train = sum(1 for s in samples if s["split"] == "train" and s["laugh"] == 1)
    laugh_val = sum(1 for s in samples if s["split"] == "val" and s["laugh"] == 1)
    laugh_test = sum(1 for s in samples if s["split"] == "test" and s["laugh"] == 1)

    print(f"Train: {laugh_train}/{train_count} laughter samples ({laugh_train/train_count*100:.1f}%)")
    print(f"Val:   {laugh_val}/{val_count} laughter samples ({laugh_val/val_count*100:.1f}%)")
    print(f"Test:  {laugh_test}/{test_count} laughter samples ({laugh_test/test_count*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Generate a laughter manifest CSV file")
    parser.add_argument("--output", type=str, default="datasets_raw/manifests/laughter_v1.csv",
                        help="Output CSV file path")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples to generate")
    parser.add_argument("--laughter-ratio", type=float, default=0.25,
                        help="Ratio of samples with laughter (0-1)")

    args = parser.parse_args()
    generate_manifest(args.output, args.samples, args.laughter_ratio)

if __name__ == "__main__":
    main()
