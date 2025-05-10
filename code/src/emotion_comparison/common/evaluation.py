#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Evaluation Framework for Emotion Feature Extraction Methods.
This module provides utilities for training classifiers and evaluating performance metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone as clone_classifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                          confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import json
import pickle
import glob
from pathlib import Path

# Define emotion labels consistently
EMOTION_NAMES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
EMOTION_SHORT = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']


class EmotionEvaluator:
    """
    Unified framework for evaluating emotion recognition methods.
    Handles loading features, training classifiers, and evaluating metrics.
    """
    
    def __init__(self, output_dir='./evaluation_results'):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dictionaries to store results
        self.features = {}
        self.labels = {}
        self.models = {}
        self.results = {}
        self.method_names = []
    
    def load_features(self, method_name, features_dir, label_map=None, normalize=True):
        """
        Load features for a specific method.
        
        Args:
            method_name: Name of the feature extraction method
            features_dir: Directory containing extracted features (.npy files)
            label_map: Function to map file names to emotion labels
            normalize: Whether to normalize features
            
        Returns:
            features: Array of features
            labels: Array of corresponding labels
        """
        print(f"Loading features for method: {method_name}")
        
        # Store the method name
        if method_name not in self.method_names:
            self.method_names.append(method_name)
        
        # List all feature files
        feature_files = glob.glob(os.path.join(features_dir, "*.npy"))
        
        if not feature_files:
            print(f"No feature files found in {features_dir}")
            return None, None
        
        print(f"Found {len(feature_files)} feature files")
        
        # Load features and extract labels
        all_features = []
        all_labels = []
        
        for file_path in tqdm(feature_files, desc=f"Loading {method_name} features"):
            try:
                # Load features
                features = np.load(file_path)
                
                # Extract filename
                base_name = os.path.basename(file_path)
                file_name = os.path.splitext(base_name)[0]
                
                # Get label
                if label_map:
                    label = label_map(file_name)
                else:
                    # Default label mapping for RAVDESS/CREMA-D
                    if '_' in file_name:
                        # CREMA-D format: 1076_IEO_ANG_XX.mp4
                        emotion_code = file_name.split('_')[2]
                        if emotion_code in ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']:
                            label = EMOTION_SHORT.index(emotion_code)
                        else:
                            continue  # Skip files with unknown emotions
                    else:
                        # RAVDESS format: 01-01-03-02-01-01-XX.mp4
                        parts = file_name.split('-')
                        if len(parts) < 3:
                            continue
                        
                        # Emotion is in the 3rd position (index 2)
                        emotion_map = {
                            '01': 4,  # neutral (NEU)
                            '02': 4,  # calm (mapped to neutral)
                            '03': 3,  # happy (HAP)
                            '04': 5,  # sad (SAD)
                            '05': 0,  # angry (ANG)
                            '06': 2,  # fearful (FEA)
                            '07': 1,  # disgust (DIS)
                            '08': 2   # surprised (mapped to fear)
                        }
                        
                        emotion_code = parts[2]
                        if emotion_code in emotion_map:
                            label = emotion_map[emotion_code]
                        else:
                            continue  # Skip files with unknown emotions
                
                # Add to collections
                all_features.append(features)
                all_labels.append(label)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not all_features:
            print(f"No valid features found for {method_name}")
            return None, None
        
        # Convert to numpy arrays
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        # Normalize features if requested
        if normalize:
            scaler = StandardScaler()
            features_array = scaler.fit_transform(features_array)
        
        # Store features and labels
        self.features[method_name] = features_array
        self.labels[method_name] = labels_array
        
        print(f"Loaded {len(features_array)} samples with shape {features_array.shape}")
        print(f"Label distribution: {np.bincount(labels_array)}")
        
        return features_array, labels_array
    
    def train_classifier(self, method_name, classifier_type='random_forest', 
                       param_grid=None, cross_val=5, n_jobs=-1):
        """
        Train a classifier on features for a specific method.
        
        Args:
            method_name: Name of the feature extraction method
            classifier_type: Type of classifier to train
            param_grid: Parameter grid for hyperparameter tuning
            cross_val: Number of cross-validation folds
            n_jobs: Number of parallel jobs for grid search
            
        Returns:
            best_model: Trained classifier with best parameters
            best_params: Best hyperparameters found
            cv_results: Cross-validation results
        """
        if method_name not in self.features:
            print(f"Features for method {method_name} not loaded. Call load_features first.")
            return None, None, None
        
        print(f"Training {classifier_type} classifier for method: {method_name}")
        
        # Get features and labels
        X = self.features[method_name]
        y = self.labels[method_name]
        
        # Select classifier and parameter grid
        if classifier_type == 'random_forest':
            classifier = RandomForestClassifier(random_state=42)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        
        elif classifier_type == 'svm':
            classifier = SVC(probability=True, random_state=42)
            if param_grid is None:
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear']
                }
        
        elif classifier_type == 'mlp':
            classifier = MLPClassifier(random_state=42, max_iter=1000)
            if param_grid is None:
                param_grid = {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
        
        elif classifier_type == 'gradient_boosting':
            classifier = GradientBoostingClassifier(random_state=42)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 5, 10]
                }
        
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            cv=cross_val,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Get best model and results
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_results = grid_search.cv_results_
        
        # Store model
        model_info = {
            'model': best_model,
            'params': best_params,
            'classifier_type': classifier_type,
            'cv_results': cv_results
        }
        
        self.models[method_name] = model_info
        
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        return best_model, best_params, cv_results
    
    def train_neural_network(self, method_name, hidden_layers=[128, 64], 
                          dropout_rate=0.5, epochs=100, batch_size=32):
        """
        Train a neural network classifier on features for a specific method.
        
        Args:
            method_name: Name of the feature extraction method
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            
        Returns:
            model: Trained neural network
            history: Training history
        """
        if method_name not in self.features:
            print(f"Features for method {method_name} not loaded. Call load_features first.")
            return None, None
        
        print(f"Training neural network for method: {method_name}")
        
        # Get features and labels
        X = self.features[method_name]
        y = self.labels[method_name]
        
        # Convert labels to one-hot encoding
        y_onehot = to_categorical(y, num_classes=len(EMOTION_NAMES))
        
        # Create model
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X.shape[1],)))
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(len(EMOTION_NAMES), activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        # Train model with cross-validation
        n_splits = 5
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_histories = []
        fold_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\nTraining fold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]
            
            # Reset model for each fold
            if fold > 0:
                model = tf.keras.models.clone_model(model)
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate on validation set
            _, val_acc = model.evaluate(X_val, y_val, verbose=0)
            fold_val_scores.append(val_acc)
            fold_histories.append(history.history)
            
            print(f"Fold {fold+1} validation accuracy: {val_acc:.4f}")
        
        # Average validation accuracy
        mean_val_acc = np.mean(fold_val_scores)
        print(f"\nMean validation accuracy: {mean_val_acc:.4f}")
        
        # Train final model on all data
        print("\nTraining final model on all data")
        
        final_model = Sequential()
        final_model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X.shape[1],)))
        final_model.add(Dropout(dropout_rate))
        
        for units in hidden_layers[1:]:
            final_model.add(Dense(units, activation='relu'))
            final_model.add(Dropout(dropout_rate))
        
        final_model.add(Dense(len(EMOTION_NAMES), activation='softmax'))
        
        final_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        final_history = final_model.fit(
            X, y_onehot,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=0.1,  # Small validation split for final training
            verbose=1
        )
        
        # Store model
        model_info = {
            'model': final_model,
            'params': {
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'batch_size': batch_size
            },
            'classifier_type': 'neural_network',
            'cv_results': {
                'fold_histories': fold_histories,
                'fold_val_scores': fold_val_scores,
                'mean_val_acc': mean_val_acc
            }
        }
        
        self.models[method_name] = model_info
        
        return final_model, final_history
    
    def evaluate_method(self, method_name, test_features=None, test_labels=None, 
                      cross_validation=True, n_splits=5):
        """
        Evaluate a trained classifier on test data or with cross-validation.
        
        Args:
            method_name: Name of the feature extraction method
            test_features: Test features (None for cross-validation)
            test_labels: Test labels (None for cross-validation)
            cross_validation: Whether to use cross-validation
            n_splits: Number of cross-validation folds
            
        Returns:
            results: Dictionary of evaluation metrics
        """
        if method_name not in self.features or method_name not in self.models:
            print(f"Method {method_name} not loaded or trained. Call load_features and train_classifier first.")
            return None
        
        print(f"Evaluating method: {method_name}")
        
        # Get features and labels
        X = self.features[method_name]
        y = self.labels[method_name]
        
        # Get model
        model_info = self.models[method_name]
        model = model_info['model']
        classifier_type = model_info['classifier_type']
        
        # Initialize results dictionary
        results = {
            'method_name': method_name,
            'classifier_type': classifier_type,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrix': None,
            'classification_report': None
        }
        
        # Evaluate on test data if provided
        if test_features is not None and test_labels is not None:
            # Make predictions
            if classifier_type == 'neural_network':
                y_pred_prob = model.predict(test_features)
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                y_pred = model.predict(test_features)
                y_pred_prob = model.predict_proba(test_features)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, y_pred)
            precision = precision_score(test_labels, y_pred, average='weighted')
            recall = recall_score(test_labels, y_pred, average='weighted')
            f1 = f1_score(test_labels, y_pred, average='weighted')
            
            # Add to results
            results['accuracy'].append(accuracy)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            
            # Compute confusion matrix and classification report
            cm = confusion_matrix(test_labels, y_pred)
            report = classification_report(test_labels, y_pred, target_names=EMOTION_NAMES)
            
            results['confusion_matrix'] = cm
            results['classification_report'] = report
            
            # Print metrics
            print(f"Test accuracy: {accuracy:.4f}")
            print(f"Test precision: {precision:.4f}")
            print(f"Test recall: {recall:.4f}")
            print(f"Test F1-score: {f1:.4f}")
            print("\nClassification Report:")
            print(report)
        
        # Cross-validation
        if cross_validation:
            # Setup cross-validation
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Initialize arrays for metrics
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            all_y_true = []
            all_y_pred = []
            
            # Perform cross-validation
            for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                if classifier_type == 'neural_network':
                    # Convert labels to one-hot encoding
                    y_train_onehot = to_categorical(y_train, num_classes=len(EMOTION_NAMES))
                    
                    # Create and train model
                    hidden_layers = model_info['params']['hidden_layers']
                    dropout_rate = model_info['params']['dropout_rate']
                    epochs = model_info['params']['epochs']
                    batch_size = model_info['params']['batch_size']
                    
                    fold_model = Sequential()
                    fold_model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X.shape[1],)))
                    fold_model.add(Dropout(dropout_rate))
                    
                    for units in hidden_layers[1:]:
                        fold_model.add(Dense(units, activation='relu'))
                        fold_model.add(Dropout(dropout_rate))
                    
                    fold_model.add(Dense(len(EMOTION_NAMES), activation='softmax'))
                    
                    fold_model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
                    ]
                    
                    fold_model.fit(
                        X_train, y_train_onehot,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        validation_split=0.1,
                        verbose=0
                    )
                    
                    # Make predictions
                    y_pred_prob = fold_model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                
                else:
                    # Clone the original model
                    fold_model = clone_classifier(model)
                    
                    # Train the model
                    fold_model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = fold_model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Add to arrays
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                
                # Collect true and predicted labels for overall confusion matrix
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                
                print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Calculate overall metrics
            overall_accuracy = np.mean(accuracies)
            overall_precision = np.mean(precisions)
            overall_recall = np.mean(recalls)
            overall_f1 = np.mean(f1_scores)
            
            # Add to results
            results['accuracy'] = accuracies
            results['precision'] = precisions
            results['recall'] = recalls
            results['f1'] = f1_scores
            
            # Compute overall confusion matrix and classification report
            cm = confusion_matrix(all_y_true, all_y_pred)
            report = classification_report(all_y_true, all_y_pred, target_names=EMOTION_NAMES)
            
            results['confusion_matrix'] = cm
            results['classification_report'] = report
            
            # Print metrics
            print(f"\nCross-Validation Results (mean ± std):")
            print(f"Accuracy: {overall_accuracy:.4f} ± {np.std(accuracies):.4f}")
            print(f"Precision: {overall_precision:.4f} ± {np.std(precisions):.4f}")
            print(f"Recall: {overall_recall:.4f} ± {np.std(recalls):.4f}")
            print(f"F1-score: {overall_f1:.4f} ± {np.std(f1_scores):.4f}")
            print("\nOverall Classification Report:")
            print(report)
        
        # Store results
        self.results[method_name] = results
        
        # Save results to disk
        self.save_results(method_name)
        
        return results
    
    def compare_methods(self, methods=None, metric='accuracy', return_df=False):
        """
        Compare multiple feature extraction methods.
        
        Args:
            methods: List of method names to compare (None for all)
            metric: Metric to compare ('accuracy', 'precision', 'recall', 'f1')
            return_df: Whether to return a DataFrame with results
            
        Returns:
            df: DataFrame with comparison results (if return_df=True)
        """
        if not methods:
            methods = self.method_names
        
        # Check if all methods have results
        for method in methods:
            if method not in self.results:
                print(f"Method {method} has no results. Call evaluate_method first.")
                return None
        
        print(f"Comparing methods using {metric}:")
        
        # Create DataFrame for results
        data = []
        
        for method in methods:
            results = self.results[method]
            metrics = results[metric]
            
            # Add to data
            data.append({
                'method': method,
                'mean': np.mean(metrics),
                'std': np.std(metrics),
                'min': np.min(metrics),
                'max': np.max(metrics),
                'classifier': results['classifier_type']
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by mean metric
        df = df.sort_values('mean', ascending=False)
        
        # Print comparison
        print(df)
        
        # Perform statistical significance testing
        self._statistical_testing(methods, metric)
        
        # Create comparison plot
        self._create_comparison_plot(methods, metric)
        
        if return_df:
            return df
    
    def _statistical_testing(self, methods, metric):
        """
        Perform statistical significance testing between methods.
        
        Args:
            methods: List of method names to compare
            metric: Metric to compare
        """
        print("\nStatistical Significance Testing:")
        
        # Create a table for p-values
        p_values = np.zeros((len(methods), len(methods)))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    p_values[i, j] = 1.0  # Same method
                else:
                    # Get metric values
                    values1 = self.results[method1][metric]
                    values2 = self.results[method2][metric]
                    
                    # Perform t-test
                    _, p_value = stats.ttest_ind(values1, values2)
                    p_values[i, j] = p_value
        
        # Create DataFrame for p-values
        p_df = pd.DataFrame(p_values, index=methods, columns=methods)
        
        # Print p-values
        print("\nP-values (row vs column):")
        print(p_df)
        
        # Highlight significant differences
        print("\nSignificant differences (p < 0.05):")
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:  # Only lower triangle
                    p_value = p_values[i, j]
                    if p_value < 0.05:
                        mean1 = np.mean(self.results[method1][metric])
                        mean2 = np.mean(self.results[method2][metric])
                        better = method1 if mean1 > mean2 else method2
                        print(f"{method1} vs {method2}: p={p_value:.4f} ({better} is better)")
    
    def _create_comparison_plot(self, methods, metric):
        """
        Create a plot comparing methods.
        
        Args:
            methods: List of method names to compare
            metric: Metric to compare
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        means = []
        stds = []
        
        for method in methods:
            values = self.results[method][metric]
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        # Create bar plot
        bars = plt.bar(methods, means, yerr=stds, capsize=10)
        
        # Customize plot
        plt.xlabel('Feature Extraction Method')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Comparison of Feature Extraction Methods ({metric.capitalize()})')
        plt.ylim(0.0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'method_comparison_{metric}.png'))
        plt.close()
        
        print(f"\nComparison plot saved to {os.path.join(self.output_dir, f'method_comparison_{metric}.png')}")
    
    def visualize_confusion_matrix(self, method_name):
        """
        Visualize the confusion matrix for a method.
        
        Args:
            method_name: Name of the feature extraction method
        """
        if method_name not in self.results:
            print(f"Method {method_name} has no results. Call evaluate_method first.")
            return
        
        # Get confusion matrix
        cm = self.results[method_name]['confusion_matrix']
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {method_name}')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{method_name}.png'))
        plt.close()
        
        print(f"Confusion matrix visualization saved to {os.path.join(self.output_dir, f'confusion_matrix_{method_name}.png')}")
    
    def save_results(self, method_name):
        """
        Save evaluation results for a method to disk.
        
        Args:
            method_name: Name of the feature extraction method
        """
        if method_name not in self.results:
            print(f"Method {method_name} has no results to save.")
            return
        
        # Create method-specific directory
        method_dir = os.path.join(self.output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # Save results as JSON
        results = self.results[method_name].copy()
        
        # Convert numpy arrays to lists for JSON serialization
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            if key in results:
                results[key] = results[key].tolist() if isinstance(results[key], np.ndarray) else results[key]
        
        # Convert confusion matrix to list
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            results['confusion_matrix'] = cm.tolist() if isinstance(cm, np.ndarray) else cm
        
        # Save to file
        with open(os.path.join(method_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save classification report as text
        if 'classification_report' in results:
            with open(os.path.join(method_dir, 'classification_report.txt'), 'w') as f:
                f.write(results['classification_report'])
        
        # Save model if available
        if method_name in self.models:
            model_info = self.models[method_name]
            
            # Save model parameters
            with open(os.path.join(method_dir, 'model_params.json'), 'w') as f:
                # Filter out non-serializable objects
                params = {k: v for k, v in model_info['params'].items() 
                       if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
                json.dump(params, f, indent=2)
            
            # Try to save the model itself if it's not a neural network
            if model_info['classifier_type'] != 'neural_network':
                try:
                    with open(os.path.join(method_dir, 'model.pkl'), 'wb') as f:
                        pickle.dump(model_info['model'], f)
                except Exception as e:
                    print(f"Could not save model: {e}")
            else:
                # Save neural network model
                try:
                    model_info['model'].save(os.path.join(method_dir, 'model.h5'))
                except Exception as e:
                    print(f"Could not save neural network model: {e}")
        
        print(f"Results for method {method_name} saved to {method_dir}")
    
    def save_all_results(self):
        """
        Save all evaluation results to disk.
        """
        for method_name in self.method_names:
            if method_name in self.results:
                self.save_results(method_name)
        
        # Save comparison results
        if len(self.method_names) > 1:
            self.compare_methods()
