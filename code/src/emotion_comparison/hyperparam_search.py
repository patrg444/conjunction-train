#!/usr/bin/env python3
"""
Hyperparameter Grid Search module for Emotion Recognition Method Comparison Framework.
Provides utilities to load hyperparameter grids and run multiple classifier configurations.
"""

import os
import json
import itertools
import numpy as np
from sklearn.model_selection import ParameterGrid
import time
from datetime import datetime

def load_param_grid(json_file):
    """
    Load hyperparameter grid configuration from a JSON file.
    
    Args:
        json_file (str): Path to JSON file containing parameter grid.
        
    Returns:
        dict: Parameter grid configuration for different classifiers.
    """
    if not os.path.exists(json_file):
        print(f"Warning: Parameter grid file not found: {json_file}")
        return {}
    
    with open(json_file, 'r') as f:
        param_grid = json.load(f)
    
    # Validate structure
    if not isinstance(param_grid, dict):
        print(f"Error: Parameter grid must be a dictionary. Found: {type(param_grid)}")
        return {}
    
    # Validate each classifier's grid
    valid_grid = {}
    for classifier, grid in param_grid.items():
        if not isinstance(grid, dict):
            print(f"Warning: Grid for classifier '{classifier}' must be a dictionary. Skipping.")
            continue
        valid_grid[classifier] = grid
    
    return valid_grid

def get_classifier_instance(classifier_name, params=None):
    """
    Get a classifier instance with the specified parameters.
    
    Args:
        classifier_name (str): Name of the classifier.
        params (dict, optional): Parameters for the classifier.
        
    Returns:
        object: Classifier instance.
    """
    if params is None:
        params = {}
    
    # Import common classifier libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Map classifier names to their classes
    classifier_map = {
        'random_forest': RandomForestClassifier,
        'svm': SVC,
        'neural_network': MLPClassifier,
        'logistic_regression': LogisticRegression,
        'knn': KNeighborsClassifier,
        'decision_tree': DecisionTreeClassifier,
        'gradient_boosting': GradientBoostingClassifier
    }
    
    if classifier_name not in classifier_map:
        print(f"Warning: Classifier '{classifier_name}' not recognized. Using RandomForestClassifier.")
        return RandomForestClassifier(**params)
    
    # Create and return classifier instance
    try:
        return classifier_map[classifier_name](**params)
    except Exception as e:
        print(f"Error creating {classifier_name} with params {params}: {str(e)}")
        print("Using default parameters instead.")
        return classifier_map[classifier_name]()

def run_grid_search(X_train, y_train, X_test, y_test, classifier_name, param_grid, output_dir):
    """
    Run grid search for a classifier with the specified parameter grid.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        classifier_name (str): Name of the classifier.
        param_grid (dict): Parameter grid for the classifier.
        output_dir (str): Directory to save results.
        
    Returns:
        tuple: (best_params, best_accuracy, all_results)
    """
    # Create results directory
    grid_results_dir = os.path.join(output_dir, f"{classifier_name}_grid_search")
    os.makedirs(grid_results_dir, exist_ok=True)
    
    # Create parameter grid
    grid = ParameterGrid(param_grid)
    print(f"Running grid search for {classifier_name} with {len(grid)} parameter combinations")
    
    # Track results
    results = []
    best_accuracy = 0
    best_params = None
    
    # Loop through all parameter combinations
    for i, params in enumerate(grid):
        start_time = time.time()
        try:
            # Get classifier instance with current parameters
            classifier = get_classifier_instance(classifier_name, params)
            
            # Train and evaluate classifier
            classifier.fit(X_train, y_train)
            accuracy = classifier.score(X_test, y_test)
            
            # Save predictions
            y_pred = classifier.predict(X_test)
            
            # Record results
            result = {
                'params': params,
                'accuracy': accuracy,
                'train_time': time.time() - start_time
            }
            results.append(result)
            
            # Update best parameters if needed
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                
                # Save predictions for best model
                np.save(os.path.join(grid_results_dir, "best_predictions.npy"), y_pred)
                np.save(os.path.join(grid_results_dir, "true_labels.npy"), y_test)
                
            # Print progress
            print(f"  - Combination {i+1}/{len(grid)}: accuracy = {accuracy:.2%}")
            
        except Exception as e:
            print(f"  - Error with parameters {params}: {str(e)}")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(grid_results_dir, f"grid_search_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'classifier': classifier_name,
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'all_results': results
        }, f, indent=2)
    
    # Save best parameters
    with open(os.path.join(grid_results_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save best accuracy
    with open(os.path.join(grid_results_dir, "accuracy.txt"), 'w') as f:
        f.write(f"{best_accuracy:.2%}\n")
    
    return best_params, best_accuracy, results

def save_grid_search_summary(output_dir, method, results_by_classifier):
    """
    Save a summary of grid search results.
    
    Args:
        output_dir (str): Directory to save results.
        method (str): Feature extraction method.
        results_by_classifier (dict): Results for each classifier.
    """
    summary_file = os.path.join(output_dir, f"{method}_grid_search_summary.json")
    
    with open(summary_file, 'w') as f:
        json.dump(results_by_classifier, f, indent=2)
    
    # Also create a human-readable text summary
    summary_txt = os.path.join(output_dir, f"{method}_grid_search_summary.txt")
    
    with open(summary_txt, 'w') as f:
        f.write(f"Grid Search Summary for {method} method\n")
        f.write("="*50 + "\n\n")
        
        # Find best overall classifier
        best_classifier = None
        best_accuracy = 0
        
        for classifier, result in results_by_classifier.items():
            accuracy = result['best_accuracy']
            params = result['best_params']
            
            f.write(f"{classifier} classifier:\n")
            f.write(f"  - Best accuracy: {accuracy:.2%}\n")
            f.write(f"  - Best parameters:\n")
            
            for param, value in params.items():
                f.write(f"    - {param}: {value}\n")
            
            f.write("\n")
            
            # Update best classifier if needed
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = classifier
        
        # Write best overall classifier
        f.write("\nBest Overall Configuration:\n")
        f.write(f"  - Classifier: {best_classifier}\n")
        f.write(f"  - Accuracy: {best_accuracy:.2%}\n")
        f.write(f"  - Parameters:\n")
        
        for param, value in results_by_classifier[best_classifier]['best_params'].items():
            f.write(f"    - {param}: {value}\n")
