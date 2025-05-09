#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

def analyze_dataset(file_path, verbose=True):
    """
    Analyze a dataset for potential issues.
    
    Args:
        file_path: Path to the dataset CSV
        verbose: Whether to print detailed analysis
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {file_path}")
    print(f"{'='*50}")
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Basic statistics
    total = len(df)
    label_counts = df['label'].value_counts().to_dict()
    label_percentage = {k: f"{v/total*100:.2f}%" for k, v in label_counts.items()}
    
    print(f"Total records: {total}")
    print(f"Label distribution: {label_counts}")
    print(f"Label percentage: {label_percentage}")
    
    # Text statistics
    df['text_length'] = df['transcript'].apply(len)
    print(f"\nText length statistics:")
    print(f"Mean: {df['text_length'].mean():.2f}")
    print(f"Min: {df['text_length'].min()}")
    print(f"Max: {df['text_length'].max()}")
    
    # Check for duplicate texts
    duplicate_texts = df.duplicated(subset=['transcript']).sum()
    print(f"\nDuplicate texts: {duplicate_texts} ({duplicate_texts/total*100:.2f}%)")
    
    # Check for most common words
    if verbose:
        print("\nMost common words:")
        all_text = ' '.join(df['transcript'].fillna(''))
        word_counts = collections.Counter(all_text.lower().split())
        for word, count in word_counts.most_common(20):
            print(f"  {word}: {count}")
    
    # Analyze most frequent starting/ending phrases
    df['first_20_chars'] = df['transcript'].fillna('').apply(lambda x: x[:20])
    df['last_20_chars'] = df['transcript'].fillna('').apply(lambda x: x[-20:] if len(x) >= 20 else x)
    
    first_chars_counts = df['first_20_chars'].value_counts().head(5)
    last_chars_counts = df['last_20_chars'].value_counts().head(5)
    
    print("\nMost common starting phrases:")
    for phrase, count in first_chars_counts.items():
        print(f"  '{phrase}': {count} times")
    
    print("\nMost common ending phrases:")
    for phrase, count in last_chars_counts.items():
        print(f"  '{phrase}': {count} times")
    
    # Check if any texts have high correlation with labels
    if verbose:
        print("\nChecking for potential data leakage...")
        try:
            vectorizer = CountVectorizer(max_features=1000)
            X = vectorizer.fit_transform(df['transcript'].fillna(''))
            y = df['label']
            
            # Train a simple baseline model to see if we get high accuracy with minimal features
            dummy_clf = DummyClassifier(strategy='stratified')
            dummy_scores = cross_val_score(dummy_clf, X, y, cv=5)
            print(f"Baseline accuracy (random guessing): {dummy_scores.mean():.4f}")
            
            # Print top features correlated with labels
            feature_names = vectorizer.get_feature_names_out()
            X_dense = X.todense()
            
            # Calculate correlation with label for each feature
            correlations = []
            for i in range(X_dense.shape[1]):
                feature_values = np.array(X_dense[:, i]).flatten()
                correlation = np.corrcoef(feature_values, y)[0, 1]
                if not np.isnan(correlation):  # Skip NaN correlations
                    correlations.append((feature_names[i], correlation))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("\nTop correlated words with labels:")
            for word, corr in correlations[:10]:
                print(f"  {word}: {corr:.4f}")
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
    
    # Check text overlap between labels
    pos_texts = set(df[df['label'] == 1]['transcript'])
    neg_texts = set(df[df['label'] == 0]['transcript'])
    overlap = pos_texts.intersection(neg_texts)
    
    print(f"\nTexts with different labels: {len(overlap)}")
    if len(overlap) > 0 and verbose:
        print("Sample overlapping texts:")
        for text in list(overlap)[:3]:
            print(f"  '{text[:50]}...'")
    
    # Print sample texts for each label
    if verbose:
        print("\nSample texts for label 0:")
        for text in df[df['label'] == 0]['transcript'].head(3):
            print(f"  '{text[:100]}...'")
        
        print("\nSample texts for label 1:")
        for text in df[df['label'] == 1]['transcript'].head(3):
            print(f"  '{text[:100]}...'")

def compare_datasets(train_path, val_path):
    """
    Compare training and validation datasets for potential issues.
    """
    print(f"\n{'='*50}")
    print(f"Comparing {train_path} and {val_path}")
    print(f"{'='*50}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    train_texts = set(train_df['transcript'])
    val_texts = set(val_df['transcript'])
    
    overlap = train_texts.intersection(val_texts)
    print(f"Text overlap between train and validation: {len(overlap)} ({len(overlap)/len(val_texts)*100:.2f}% of validation)")
    
    if len(overlap) > 0:
        print("\nSample overlapping texts:")
        for text in list(overlap)[:3]:
            # Find the labels for this text in both datasets
            train_label = train_df[train_df['transcript'] == text]['label'].values[0]
            val_label = val_df[val_df['transcript'] == text]['label'].values[0]
            print(f"  '{text[:50]}...' (Train label: {train_label}, Val label: {val_label})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze humor datasets for potential issues")
    parser.add_argument("--train", default="datasets/manifests/humor/train_humor_with_text.csv", 
                        help="Path to training dataset")
    parser.add_argument("--val", default="datasets/manifests/humor/val_humor_with_text.csv",
                        help="Path to validation dataset")
    parser.add_argument("--verbose", action="store_true", help="Print detailed analysis")
    
    args = parser.parse_args()
    
    # Analyze individual datasets
    analyze_dataset(args.train, args.verbose)
    analyze_dataset(args.val, args.verbose)
    
    # Compare datasets
    compare_datasets(args.train, args.val)
