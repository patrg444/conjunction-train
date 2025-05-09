#!/usr/bin/env python3
"""
Evaluate the WAV2VEC attention model on the full dataset to test generalization.
This script:
1. Loads all available WAV2VEC feature files
2. Loads the trained model
3. Evaluates the model on the full dataset
4. Computes metrics and generates a confusion matrix visualization
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate WAV2VEC model across the dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--label-classes', type=str, required=True, help='Path to the label classes file')
    parser.add_argument('--feature-dir', type=str, default='/home/ubuntu/wav2vec_sample/home/ubuntu/audio_emotion/models/wav2vec',
                        help='Directory with WAV2VEC feature files')
    parser.add_argument('--max-length', type=int, default=221, help='Maximum sequence length for padding')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--cross-validation', action='store_true', help='Perform k-fold cross-validation')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for cross-validation')
    return parser.parse_args()

def load_wav2vec_features(feature_dir):
    """Load all WAV2VEC feature files from the directory"""
    print(f"Searching in directory: {feature_dir}")
    feature_files = glob.glob(os.path.join(feature_dir, "*.npz"))
    print(f"Found {len(feature_files)} feature files in {feature_dir}")
    
    features = []
    labels = []
    file_ids = []
    
    for file_path in feature_files:
        try:
            # Extract file ID from path
            file_id = os.path.basename(file_path).split('.')[0]
            
            # Load the NPZ file
            data = np.load(file_path, allow_pickle=True)
            
            # Check for different possible key names for embeddings
            if 'wav2vec_features' in data:
                embedding = data['wav2vec_features']
            elif 'embeddings' in data:
                embedding = data['embeddings']
            else:
                print(f"Error loading {file_path}: 'wav2vec_features' or 'embeddings' is not a file in the archive")
                continue
                
            label = data['label']
            
            features.append(embedding)
            labels.append(label)
            file_ids.append(file_id)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not features:
        print("No features loaded. Exiting.")
        return None, None, None
    
    print(f"Successfully loaded {len(features)} feature files.")
    return features, np.array(labels), file_ids

def preprocess_features(features, max_len):
    """Pad features to the same length"""
    padded_features = []
    for feature in features:
        if len(feature) > max_len:
            # Truncate
            padded_feature = feature[:max_len]
        else:
            # Pad with zeros
            padding = np.zeros((max_len - len(feature), feature.shape[1]))
            padded_feature = np.vstack((feature, padding))
        padded_features.append(padded_feature)
    
    return np.array(padded_features)

def evaluate_model(model, X, y, class_names):
    """Evaluate model performance on the dataset"""
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_true_classes)
    balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    
    # Create results dictionary
    results = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return results, cm_normalized, y_true_classes, y_pred_classes

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save the confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def perform_cross_validation(features, labels, max_len, n_folds, class_names, output_dir):
    """Perform k-fold cross-validation on the dataset"""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_indices = np.argmax(labels, axis=1)
    
    fold_results = []
    all_true = []
    all_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.zeros(len(features)), y_indices)):
        print(f"\nFold {fold+1}/{n_folds}")
        
        # Split the data
        X_train = [features[i] for i in train_idx]
        y_train = labels[train_idx]
        X_test = [features[i] for i in test_idx]
        y_test = labels[test_idx]
        
        # Preprocess
        X_train_padded = preprocess_features(X_train, max_len)
        X_test_padded = preprocess_features(X_test, max_len)
        
        # Build a simple model for this fold
        model = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(64)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(y_train.shape[1], activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        model.fit(
            X_train_padded, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test_padded)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Collect results
        all_true.extend(y_true_classes)
        all_pred.extend(y_pred_classes)
        
        # Calculate fold metrics
        fold_acc = np.mean(y_pred_classes == y_true_classes)
        fold_balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
        fold_f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': float(fold_acc),
            'balanced_accuracy': float(fold_balanced_acc),
            'f1_score': float(fold_f1)
        })
        
        print(f"Fold {fold+1} Results:")
        print(f"  Accuracy: {fold_acc:.4f}")
        print(f"  Balanced Accuracy: {fold_balanced_acc:.4f}")
        print(f"  F1 Score: {fold_f1:.4f}")
    
    # Calculate overall metrics
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    overall_acc = np.mean(all_true == all_pred)
    overall_balanced_acc = balanced_accuracy_score(all_true, all_pred)
    overall_f1 = f1_score(all_true, all_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm_normalized, 
        class_names, 
        os.path.join(output_dir, 'cv_confusion_matrix.png')
    )
    
    # Create results dictionary
    cv_results = {
        'folds': fold_results,
        'overall': {
            'accuracy': float(overall_acc),
            'balanced_accuracy': float(overall_balanced_acc),
            'f1_score': float(overall_f1),
            'confusion_matrix': cm.tolist()
        }
    }
    
    # Save results
    with open(os.path.join(output_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print("\nCross-Validation Overall Results:")
    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  Balanced Accuracy: {overall_balanced_acc:.4f}")
    print(f"  F1 Score: {overall_f1:.4f}")
    
    # Also save the label classes used
    with open(os.path.join(output_dir, 'label_classes.npy'), 'wb') as f:
        np.save(f, np.array(class_names))
    
    return cv_results

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load WAV2VEC features
    features, labels, file_ids = load_wav2vec_features(args.feature_dir)
    if features is None:
        return
    
    # Load label classes
    class_names = np.load(args.label_classes)
    
    if args.cross_validation:
        # Perform cross-validation
        cv_results = perform_cross_validation(
            features, labels, args.max_length, args.n_folds, class_names, args.output_dir
        )
        print("Cross-validation complete. Results saved to:", args.output_dir)
    else:
        # Load the trained model
        model = keras.models.load_model(args.model_path)
        
        # Preprocess features
        X = preprocess_features(features, args.max_length)
        
        # Evaluate model
        results, cm_normalized, y_true, y_pred = evaluate_model(model, X, labels, class_names)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            cm_normalized, 
            class_names, 
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        # Save results
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nEvaluation Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"  F1 Score: {results['f1_score']:.4f}")
        print("Results saved to:", args.output_dir)

if __name__ == "__main__":
    main()
