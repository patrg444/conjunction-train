#!/usr/bin/env python3
"""
Emotion Recognition Method Comparison Framework
Main script for running comparisons between different feature extraction methods
"""

import argparse
import os
import sys
import time
import numpy as np
import json

# Import our visualization module
from common.visualization import (
    generate_html_report,
    generate_confusion_matrix_reports,
    plot_accuracy_comparison,
    fig_to_base64
)

# Import hyperparameter search module
from hyperparam_search import (
    load_param_grid,
    run_grid_search,
    save_grid_search_summary
)

def parse_args():
    parser = argparse.ArgumentParser(description='Emotion Recognition Method Comparison')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--methods', type=str, required=True, help='Feature extraction methods (space-separated list)')
    parser.add_argument('--classifiers', type=str, required=True, help='Classifiers to use (space-separated list)')
    parser.add_argument('--extract_features', action='store_true', help='Extract features from videos')
    parser.add_argument('--resume_extraction', action='store_true', help='Resume feature extraction')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--cross_validate', action='store_true', help='Perform cross-validation')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--max_videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--dataset_name', type=str, default='RAVDESS', help='Name of the dataset for reporting')
    parser.add_argument('--param_grid_json', type=str, help='JSON file with hyperparameter grid search specifications')
    parser.add_argument('--grid_search', action='store_true', help='Perform hyperparameter grid search using param_grid_json')

    return parser.parse_args()

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def simulate_feature_extraction(args, method):
    """Simulate feature extraction for demonstration."""
    print(f"[Demo] Extracting features using {method} method...")

    # Simulate processing time
    num_videos = args.max_videos if args.max_videos else 20
    for i in range(0, num_videos, 5):
        print(f"  - Processed {i}/{num_videos} videos...")
        time.sleep(0.1)  # Simulate processing time
    
    # Make sure we report the last batch
    if num_videos % 5 != 0 or num_videos - 5 != i:
        print(f"  - Processed {num_videos}/{num_videos} videos...")

    print(f"[Demo] Completed feature extraction with {method} method.")

    # Create a dummy feature file in the output directory
    feature_dir = os.path.join(args.output_dir, f"{method}_features")
    create_directory(feature_dir)
    with open(os.path.join(feature_dir, "features.txt"), "w") as f:
        f.write(f"Demo features extracted with {method} method\n")
        f.write(f"For {num_videos} videos from {args.video_dir}\n")

    return feature_dir

def simulate_classifier_training(args, method, classifier, param_grid=None):
    """Simulate classifier training for demonstration."""
    print(f"[Demo] Training {classifier} classifier on {method} features...")

    # Set up directories
    features_dir = os.path.join(args.output_dir, f"{method}_features")
    results_dir = os.path.join(args.output_dir, f"{method}_{classifier}_results")
    create_directory(results_dir)

    # Generate synthetic data for visualization
    num_classes = 8  # Common for emotion datasets (e.g., 8 emotions in RAVDESS)
    num_samples = 100
    
    # Simulated true labels and predictions
    np.random.seed(hash(method + classifier) % 10000)  # Use hash for reproducibility
    y_true = np.random.randint(0, num_classes, num_samples)
    
    # Generate simulated feature data
    feature_dim = 128  # Typical dimension for emotion features
    X = np.random.normal(0, 1, (num_samples, feature_dim))
    
    # Split data for training/testing
    train_ratio = 0.8
    train_size = int(train_ratio * num_samples)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_true[:train_size], y_true[train_size:]
    
    # If grid search is enabled and param grid is provided
    if args.grid_search and param_grid and classifier in param_grid:
        print(f"[Demo] Performing grid search for {classifier} with {len(param_grid[classifier])} parameters...")
        best_params, best_accuracy, _ = run_grid_search(
            X_train, y_train, X_test, y_test, 
            classifier, param_grid[classifier], results_dir
        )
        accuracy = best_accuracy * 100  # convert to percentage
        print(f"[Demo] Best parameters: {best_params}")
        print(f"[Demo] Best accuracy: {accuracy:.2f}%")
        
        # Generate predictions with best parameters
        from hyperparam_search import get_classifier_instance
        best_classifier = get_classifier_instance(classifier, best_params)
        best_classifier.fit(X_train, y_train)
        y_pred = best_classifier.predict(X_test)
    else:
        # Make predictions somewhat correlated with true labels to simulate realistic accuracy
        confusion_factor = 0.2  # How much confusion to introduce
        random_shift = np.random.normal(0, confusion_factor, len(y_test)).astype(int)
        y_pred = np.clip(y_test + random_shift, 0, num_classes - 1)
        
        # Simulate training time
        accuracies = []
        for i in range(5):  # 5 epochs
            # Start with lower accuracy and improve
            accuracy = 50 + i*10 + (hash(method) % 5) + (hash(classifier) % 5)
            accuracy = min(accuracy, 95)  # Cap at 95%
            accuracies.append(accuracy / 100)
            print(f"  - Epoch {i+1}: accuracy = {accuracy:.2f}%")
            time.sleep(0.2)  # Simulate training time
        
        # Save accuracy
        with open(os.path.join(results_dir, "accuracy.txt"), "w") as f:
            f.write(f"{accuracy:.2f}%\n")
    
    # Save predictions and true labels for confusion matrix visualization
    np.save(os.path.join(results_dir, "predictions.npy"), y_pred)
    np.save(os.path.join(results_dir, "true_labels.npy"), y_test)
    
    # Save class names
    class_names = [f"Emotion_{i+1}" for i in range(num_classes)]
    with open(os.path.join(results_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    return accuracy

def main():
    args = parse_args()

    # Parse methods and classifiers (split by space)
    methods = args.methods.split()
    classifiers = args.classifiers.split()

    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Methods: {methods}")
    print(f"Classifiers: {classifiers}")

    # Create output directory
    create_directory(args.output_dir)
    
    # Load hyperparameter grid if specified
    param_grid = None
    if args.param_grid_json:
        print(f"Loading hyperparameter grid from: {args.param_grid_json}")
        param_grid = load_param_grid(args.param_grid_json)
        if not param_grid:
            print("Warning: No valid hyperparameter grid found. Proceeding with default parameters.")
            args.grid_search = False

    # Extract features if requested
    if args.extract_features:
        print("\n=== Feature Extraction Phase ===")
        for method in methods:
            feature_dir = simulate_feature_extraction(args, method)
            print(f"Features saved to: {feature_dir}")

    # Train and evaluate classifiers
    print("\n=== Classification Phase ===")
    results = {}
    for method in methods:
        results[method] = {}
        
        # Track grid search results if applicable
        if args.grid_search and param_grid:
            grid_search_results = {}
        
        for classifier in classifiers:
            accuracy = simulate_classifier_training(args, method, classifier, param_grid)
            results[method][classifier] = accuracy
            
            # Store grid search results if applicable
            if args.grid_search and param_grid and classifier in param_grid:
                grid_search_dir = os.path.join(args.output_dir, f"{method}_{classifier}_results")
                best_params_file = os.path.join(grid_search_dir, "best_params.json")
                
                if os.path.exists(best_params_file):
                    with open(best_params_file, 'r') as f:
                        best_params = json.load(f)
                    
                    grid_search_results[classifier] = {
                        'best_params': best_params,
                        'best_accuracy': accuracy / 100  # Convert percentage to fraction
                    }
        
        # Save grid search summary if applicable
        if args.grid_search and param_grid and 'grid_search_results' in locals():
            save_grid_search_summary(args.output_dir, method, grid_search_results)

    # Create summary
    print("\n=== Summary ===")
    with open(os.path.join(args.output_dir, "comparison_summary.txt"), "w") as f:
        f.write("Emotion Recognition Method Comparison Summary\n\n")

        # Find best method-classifier combination
        best_accuracy = 0
        best_method = None
        best_classifier = None

        for method in methods:
            f.write(f"{method} method results:\n")
            for classifier in classifiers:
                accuracy = results[method][classifier]
                f.write(f"  - {classifier}: {accuracy:.2f}%\n")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_method = method
                    best_classifier = classifier
            f.write("\n")

        # Write best-performing method
        f.write("The best-performing method was:\n")
        f.write(f"  {best_method} features with {best_classifier} classifier\n")
        f.write(f"  Accuracy: {best_accuracy:.2f}%\n")
        
        # Add grid search info if applicable
        if args.grid_search and param_grid:
            f.write("\nGrid search was performed with the following parameters:\n")
            for classifier in classifiers:
                if classifier in param_grid:
                    f.write(f"  - {classifier}: {list(param_grid[classifier].keys())}\n")

    # Generate visualizations and reports
    if args.visualize:
        print("\n=== Generating Visualizations ===")
        
        # Generate confusion matrices
        cm_dir = generate_confusion_matrix_reports(args.output_dir, args.dataset_name)
        print(f"Confusion matrices saved to: {cm_dir}")
        
        # Generate HTML report
        report_path = generate_html_report(args.output_dir, args.dataset_name)
        print(f"HTML report saved to: {report_path}")
        
        # Generate extra visualizations for grid search if applicable
        if args.grid_search and param_grid:
            print("Generating grid search visualizations...")
            for method in methods:
                grid_summary_file = os.path.join(args.output_dir, f"{method}_grid_search_summary.txt")
                if os.path.exists(grid_summary_file):
                    print(f"Grid search summary saved to: {grid_summary_file}")

    print(f"Comparison complete! Results saved to: {args.output_dir}")
    print(f"Best method: {best_method} features with {best_classifier} classifier ({best_accuracy:.2f}%)")

if __name__ == "__main__":
    main()
