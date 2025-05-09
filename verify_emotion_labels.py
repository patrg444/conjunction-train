#!/usr/bin/env python3
"""
Verify emotion labels in WAV2VEC feature files.
This script:
1. Searches for WAV2VEC feature files
2. Extracts and analyzes emotion labels
3. Identifies any inconsistencies or imbalances
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse


def find_feature_files(search_dirs=None):
    """Search for WAV2VEC feature files in multiple possible locations"""
    if search_dirs is None:
        # Default search locations
        search_dirs = [
            "/home/ubuntu/audio_emotion/wav2vec_features",
            "/home/ubuntu/wav2vec_features",
            "/home/ubuntu/audio_emotion/features/wav2vec",
            "/home/ubuntu/features/wav2vec",
            "/data/wav2vec_features",
            "/home/ubuntu/wav2vec_sample/home/ubuntu/audio_emotion/models/wav2vec"
        ]
    
    # Try all specified directories
    feature_files = []
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            print(f"Searching in directory: {dir_path}")
            npz_files = glob.glob(os.path.join(dir_path, "*.npz"))
            if npz_files:
                feature_files.extend(npz_files)
                print(f"Found {len(npz_files)} feature files in {dir_path}")
    
    # If no files found, try a wider search
    if not feature_files:
        print("No feature files found in specified directories. Trying a wider search...")
        npz_files = glob.glob(os.path.join("/home/ubuntu", "**/*.npz"), recursive=True)
        if npz_files:
            feature_files.extend(npz_files)
            print(f"Found {len(npz_files)} feature files in wider search")
    
    return feature_files


def extract_labels(feature_files):
    """Extract labels from feature files"""
    labels = []
    filenames = []
    
    for file_path in feature_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Try to extract label from file content
            if 'label' in data:
                label = data['label']
            else:
                # Extract label from filename if not in file content
                # Typically the 3rd component in filenames like "Actor_01_01_01_01_anger.npz"
                filename = os.path.basename(file_path)
                components = filename.split('_')
                if len(components) >= 3:
                    # Check if the 3rd component is the emotion label
                    possible_label = components[2]
                    if possible_label in ['01', '02', '03', '04', '05', '06', '07', '08']:
                        # Convert numerical code to emotion name if it matches known patterns
                        label_map = {
                            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
                        }
                        label = label_map.get(possible_label, possible_label)
                    else:
                        # Try to extract from the last part (removing .npz extension)
                        last_part = components[-1].split('.')[0]
                        label = last_part
                else:
                    # Default if we can't parse
                    label = "unknown"
            
            labels.append(str(label))
            filenames.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error extracting label from {file_path}: {str(e)}")
    
    return labels, filenames


def analyze_labels(labels, filenames):
    """Analyze extracted labels for distribution and consistency"""
    # Count label occurrences
    label_counts = Counter(labels)
    
    print("\nEmotion Label Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")
    
    # Check for minimum count per class (for stratification)
    min_count = min(label_counts.values()) if label_counts else 0
    print(f"\nMinimum samples per class: {min_count}")
    
    # Identify any problematic or inconsistent labels
    # These could be misspellings, variations, etc.
    similar_labels = []
    known_emotion_stems = [
        'anger', 'happy', 'sad', 'fear', 'disgust', 'surprise', 'neutral', 'calm'
    ]
    
    for label in label_counts:
        for stem in known_emotion_stems:
            if stem in label.lower() and label.lower() != stem:
                similar_labels.append((label, stem))
    
    if similar_labels:
        print("\nPossible label inconsistencies detected:")
        for label, stem in similar_labels:
            print(f"  '{label}' might be a variant of '{stem}'")
    
    # Check for unknown or non-standard labels
    standard_emotions = [
        'anger', 'angry', 'happiness', 'happy', 'sad', 'sadness', 
        'fear', 'fearful', 'disgust', 'surprised', 'surprise', 
        'neutral', 'calm', '01', '02', '03', '04', '05', '06', '07', '08'
    ]
    
    unknown_labels = [l for l in label_counts if l.lower() not in [s.lower() for s in standard_emotions]]
    if unknown_labels:
        print("\nUnknown or non-standard labels detected:")
        for label in unknown_labels:
            print(f"  '{label}': {label_counts[label]} samples")
            # Print a few example filenames
            examples = [f for l, f in zip(labels, filenames) if l == label][:3]
            print(f"  Examples: {', '.join(examples)}")
    
    return label_counts


def plot_distribution(label_counts, output_file="label_distribution.png"):
    """Plot the label distribution"""
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    labels = [labels[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    plt.bar(labels, counts)
    plt.title('Emotion Label Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add counts as text labels on the bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    
    plt.savefig(output_file)
    print(f"\nDistribution plot saved to {output_file}")
    return output_file


def check_label_mapping(labels, label_classes_file=None):
    """Check if the extracted labels match with the saved label classes"""
    if label_classes_file and os.path.exists(label_classes_file):
        try:
            label_classes = np.load(label_classes_file, allow_pickle=True)
            print(f"\nLoaded {len(label_classes)} label classes from {label_classes_file}")
            
            # Check if all classes in the file are present in the extracted labels
            missing_in_data = [c for c in label_classes if c not in labels]
            if missing_in_data:
                print(f"Warning: The following classes from {label_classes_file} are not found in the data:")
                for c in missing_in_data:
                    print(f"  {c}")
            
            # Check if all extracted labels are in the label classes file
            missing_in_file = [l for l in set(labels) if l not in label_classes]
            if missing_in_file:
                print(f"Warning: The following labels from the data are not in {label_classes_file}:")
                for l in missing_in_file:
                    print(f"  {l}")
            
            if not missing_in_data and not missing_in_file:
                print("All labels match between the data and the label classes file.")
            
            return label_classes
        except Exception as e:
            print(f"Error loading label classes: {str(e)}")
    
    # If no file provided or error, create a new mapping
    unique_labels = sorted(list(set(labels)))
    print("\nConstructed label mapping from data:")
    for i, label in enumerate(unique_labels):
        print(f"  {i}: {label}")
    
    return np.array(unique_labels)


def main():
    parser = argparse.ArgumentParser(description='Verify WAV2VEC emotion labels')
    parser.add_argument('--search-dir', type=str, nargs='+',
                        help='Directory to search for feature files')
    parser.add_argument('--label-classes', type=str, default=None,
                        help='Path to the label classes file for comparison')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find feature files
    feature_files = find_feature_files(args.search_dir)
    
    if not feature_files:
        print("No feature files found. Exiting.")
        return
    
    print(f"Found {len(feature_files)} feature files.")
    
    # Extract labels
    labels, filenames = extract_labels(feature_files)
    
    if not labels:
        print("No labels extracted. Exiting.")
        return
    
    print(f"Extracted {len(labels)} labels.")
    
    # Analyze labels
    label_counts = analyze_labels(labels, filenames)
    
    # Check label mapping
    label_classes = check_label_mapping(labels, args.label_classes)
    
    # Save the label classes
    output_file = os.path.join(args.output_dir, 'verified_label_classes.npy')
    np.save(output_file, label_classes)
    print(f"Saved verified label classes to {output_file}")
    
    # Plot distribution
    plot_path = os.path.join(args.output_dir, 'label_distribution.png')
    plot_distribution(label_counts, plot_path)
    
    # Save distribution to CSV
    csv_path = os.path.join(args.output_dir, 'label_distribution.csv')
    with open(csv_path, 'w') as f:
        f.write("label,count\n")
        for label, count in sorted(label_counts.items()):
            f.write(f"{label},{count}\n")
    print(f"Saved label distribution to {csv_path}")
    
    print("\nLabel verification complete.")


if __name__ == "__main__":
    main()
