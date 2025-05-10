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

def load_wav2vec_features(feature_files):
    """Load all WAV2VEC features and their corresponding labels from npz files"""
    all_features = []
    all_labels = []
    all_filenames = []
    
    for file_path in feature_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            features = data['features'] if 'features' in data else data['embeddings']
            label = data['label'] if 'label' in data else os.path.basename(file_path).split('_')[2]
            
            all_features.append(features)
            all_labels.append(label)
            all_filenames.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return all_features, all_labels, all_filenames

def pad_sequences(features, max_length=None):
    """Pad sequences to a fixed length"""
    if max_length is None:
        # Calculate sequence lengths and set padding length to 95th percentile
        seq_lengths = [len(f) for f in features]
        max_length = int(np.percentile(seq_lengths, 95))
    
    # Print sequence length statistics
    seq_lengths = [len(f) for f in features]
    print(f"Sequence length statistics:")
    print(f"  Min: {min(seq_lengths)}")
    print(f"  Max: {max(seq_lengths)}")
    print(f"  Mean: {np.mean(seq_lengths):.2f}")
    print(f"  Median: {np.median(seq_lengths)}")
    print(f"  95th percentile: {np.percentile(seq_lengths, 95)}")
    
    print(f"Padding sequences to length {max_length}")
    
    # Pad sequences
    padded_features = []
    for feature in features:
        if len(feature) > max_length:
            # Truncate
            padded_feature = feature[:max_length]
        else:
            # Pad with zeros
            padding = np.zeros((max_length - len(feature), feature.shape[1]))
            padded_feature = np.vstack((feature, padding))
        padded_features.append(padded_feature)
    
    return np.array(padded_features)

def encode_labels(labels, label_classes=None):
    """Convert string labels to one-hot encoded vectors"""
    if label_classes is None:
        # Get unique labels
        unique_labels = list(set(labels))
        unique_labels.sort()
        label_classes = np.array(unique_labels)
    
    # Map labels to indices
    label_to_index = {label: i for i, label in enumerate(label_classes)}
    
    # Convert labels to indices
    indices = np.array([label_to_index.get(label, -1) for label in labels])
    
    # Check for unknown labels
    unknown_indices = np.where(indices == -1)[0]
    if len(unknown_indices) > 0:
        unknown_labels = [labels[i] for i in unknown_indices]
        print(f"Warning: Found {len(unknown_indices)} unknown labels: {unknown_labels}")
        # Replace unknown labels with a valid index (0)
        indices[unknown_indices] = 0
    
    # One-hot encode
    n_classes = len(label_classes)
    one_hot = np.zeros((len(indices), n_classes))
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1
    
    return one_hot, indices, label_classes

def load_model(model_path):
    """Load the trained model"""
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"Model loaded from {model_path}")
        # Compile with same metrics to evaluate
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def plot_confusion_matrix(y_true, y_pred, class_names, output_file="confusion_matrix.png"):
    """Generate and save a confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add values to the plot
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_file}")
    return cm

def compute_class_metrics(y_true, y_pred, class_names):
    """Compute per-class accuracy and F1 score"""
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Per-class F1 score
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Combine metrics
    class_metrics = {
        class_name: {
            'accuracy': per_class_acc[i],
            'f1': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
        for i, class_name in enumerate(class_names)
    }
    
    return class_metrics

def run_cross_validation(features, labels, label_classes, n_splits=5, max_length=221):
    """Perform stratified K-fold cross-validation"""
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    # Initialize K-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Metrics to collect
    fold_results = []
    all_true = []
    all_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Split data
        X_train = np.array([features[i] for i in train_idx])
        X_test = np.array([features[i] for i in test_idx])
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]
        
        # Pad sequences
        X_train_padded = pad_sequences(X_train, max_length)
        X_test_padded = pad_sequences(X_test, max_length)
        
        # Encode labels
        y_train_one_hot, _, _ = encode_labels(y_train, label_classes)
        y_test_one_hot, y_test_indices, _ = encode_labels(y_test, label_classes)
        
        # Create a simple model for testing
        # You would typically load your pre-trained model architecture and just retrain
        # This is a placeholder for demonstration
        input_shape = (X_train_padded.shape[1], X_train_padded.shape[2])
        
        inputs = keras.layers.Input(shape=input_shape)
        mask = keras.layers.Masking(mask_value=0.0)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(mask)
        attention = keras.layers.Dense(1, activation='tanh')(lstm)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation('softmax')(attention)
        attention = keras.layers.RepeatVector(256)(attention)
        attention = keras.layers.Permute([2, 1])(attention)
        context = keras.layers.Multiply()([lstm, attention])
        context = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(context)
        dense = keras.layers.Dense(128, activation='relu')(context)
        outputs = keras.layers.Dense(len(label_classes), activation='softmax')(dense)
        
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Train the model
        model.fit(
            X_train_padded, y_train_one_hot,
            epochs=10,  # Reduced for testing
            batch_size=32,
            validation_data=(X_test_padded, y_test_one_hot),
            verbose=1
        )
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_padded, y_test_one_hot, verbose=0)
        
        # Predictions
        y_pred_proba = model.predict(X_test_padded)
        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        
        # Add to all predictions for overall metrics
        all_true.extend(y_test_indices)
        all_pred.extend(y_pred_indices)
        
        # Calculate additional metrics
        balanced_acc = balanced_accuracy_score(y_test_indices, y_pred_indices)
        macro_f1 = f1_score(y_test_indices, y_pred_indices, average='macro')
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1
        })
        
        print(f"Fold {fold+1} results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
    
    # Compute overall metrics across all folds
    overall_accuracy = np.mean([res['accuracy'] for res in fold_results])
    overall_balanced_acc = np.mean([res['balanced_accuracy'] for res in fold_results])
    overall_macro_f1 = np.mean([res['macro_f1'] for res in fold_results])
    
    print("\nCross-validation overall results:")
    print(f"  Accuracy: {overall_accuracy:.4f} ± {np.std([res['accuracy'] for res in fold_results]):.4f}")
    print(f"  Balanced Accuracy: {overall_balanced_acc:.4f} ± {np.std([res['balanced_accuracy'] for res in fold_results]):.4f}")
    print(f"  Macro F1: {overall_macro_f1:.4f} ± {np.std([res['macro_f1'] for res in fold_results]):.4f}")
    
    # Generate overall confusion matrix from all folds
    cm = confusion_matrix(all_true, all_pred)
    
    cv_results = {
        'fold_results': fold_results,
        'overall': {
            'accuracy': overall_accuracy,
            'balanced_accuracy': overall_balanced_acc,
            'macro_f1': overall_macro_f1
        }
    }
    
    return cv_results, cm

def main():
    parser = argparse.ArgumentParser(description='Evaluate WAV2VEC model on the full dataset')
    parser.add_argument('--model-path', type=str, default='models/wav2vec_v9_attention/best_model_v9.h5',
                        help='Path to the trained model')
    parser.add_argument('--label-classes', type=str, default='models/wav2vec_v9_attention/label_classes_v9.npy',
                        help='Path to the label classes file')
    parser.add_argument('--max-length', type=int, default=221,
                        help='Maximum sequence length for padding')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--search-dir', type=str, nargs='+',
                        help='Directory to search for feature files')
    parser.add_argument('--cross-validation', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find feature files
    feature_files = find_feature_files(args.search_dir)
    
    if not feature_files:
        print("No feature files found. Exiting.")
        return
    
    print(f"Found {len(feature_files)} feature files.")
    
    # Load features
    features, labels, filenames = load_wav2vec_features(feature_files)
    
    if not features:
        print("No features loaded. Exiting.")
        return
    
    print(f"Loaded {len(features)} features with {len(set(labels))} unique classes.")
    
    # Check for label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} samples")
    
    # Load label classes if file exists
    label_classes = None
    if os.path.exists(args.label_classes):
        try:
            label_classes = np.load(args.label_classes, allow_pickle=True)
            print(f"Loaded {len(label_classes)} label classes from {args.label_classes}")
        except Exception as e:
            print(f"Error loading label classes: {str(e)}")
    
    # Encode labels
    labels_one_hot, label_indices, label_classes = encode_labels(labels, label_classes)
    
    # Save the label classes
    np.save(os.path.join(args.output_dir, 'label_classes.npy'), label_classes)
    print(f"Saved label classes to {os.path.join(args.output_dir, 'label_classes.npy')}")
    
    # Print label mapping
    print("\nLabel mapping:")
    for i, label in enumerate(label_classes):
        print(f"  {i}: {label}")
    
    # Pad sequences
    padded_features = pad_sequences(features, args.max_length)
    
    print(f"Padded features shape: {padded_features.shape}")
    
    if args.cross_validation:
        # Perform cross-validation
        cv_results, cv_cm = run_cross_validation(
            features, labels, label_classes, n_splits=args.n_folds, max_length=args.max_length
        )
        
        # Save cross-validation results
        with open(os.path.join(args.output_dir, 'cross_validation_results.json'), 'w') as f:
            # Convert NumPy types to Python native types for JSON serialization
            cv_results_serializable = json.loads(json.dumps(cv_results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x))
            json.dump(cv_results_serializable, f, indent=2)
        
        # Plot and save confusion matrix from cross-validation
        plt.figure(figsize=(12, 10))
        plt.imshow(cv_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Cross-Validation Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(label_classes))
        plt.xticks(tick_marks, label_classes, rotation=45, ha='right')
        plt.yticks(tick_marks, label_classes)
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(args.output_dir, 'cv_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    else:
        # Try to load the model
        model = load_model(args.model_path)
        
        if model is None:
            print("Could not load model. Exiting.")
            return
        
        # Evaluate on full dataset
        print("\nEvaluating on full dataset...")
        loss, accuracy = model.evaluate(padded_features, labels_one_hot)
        
        print(f"\nFull dataset evaluation results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Generate predictions
        y_pred_proba = model.predict(padded_features)
        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        
        # Calculate additional metrics
        balanced_acc = balanced_accuracy_score(label_indices, y_pred_indices)
        macro_f1 = f1_score(label_indices, y_pred_indices, average='macro')
        
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        
        # Generate confusion matrix
        cm = plot_confusion_matrix(
            label_indices, y_pred_indices, label_classes,
            output_file=os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        # Compute per-class metrics
        class_metrics = compute_class_metrics(label_indices, y_pred_indices, label_classes)
        
        # Print per-class metrics
        print("\nPer-class metrics:")
        for class_name, metrics in class_metrics.items():
            print(f"  {class_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")
            print(f"    Support: {metrics['support']}")
        
        # Save evaluation results
        results = {
            'overall': {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'macro_f1': float(macro_f1)
            },
            'class_metrics': class_metrics
        }
        
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            # Convert NumPy types to Python native types for JSON serialization
            results_serializable = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x))
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to {os.path.join(args.output_dir, 'evaluation_results.json')}")

if __name__ == "__main__":
    main()
