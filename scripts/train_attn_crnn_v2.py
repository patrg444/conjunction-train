#!/usr/bin/env python3
import os
import glob
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LayerNormalization, Attention, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import time
import shutil

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class WavDataGenerator(tf.keras.utils.Sequence):
    """Data generator with augmentation for wav2vec features."""
    def __init__(self, X, y, batch_size=32, shuffle=True, 
                 apply_noise=True, noise_sigma=0.005,
                 apply_mixup=True, mixup_alpha=0.2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        
        # Augmentation flags
        self.apply_noise = apply_noise
        self.noise_sigma = noise_sigma
        self.apply_mixup = apply_mixup
        self.mixup_alpha = mixup_alpha
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generates one batch of data."""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        # Apply augmentations
        if self.apply_noise:
            # Add Gaussian noise to wav2vec features
            noise = np.random.normal(0, self.noise_sigma, batch_X.shape)
            batch_X = batch_X + noise
        
        if self.apply_mixup and len(batch_X) > 1:
            # Apply MixUp augmentation
            batch_X, batch_y = self._mixup(batch_X, batch_y)
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        """Updates indices after each epoch."""
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _mixup(self, X, y):
        """Apply MixUp augmentation to the batch."""
        # Generate mixup parameters
        batch_size = len(X)
        weights = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size)
        weights = np.maximum(weights, 1 - weights)
        weights = weights.reshape(batch_size, 1, 1)
        
        # Create shuffled indices
        indices = np.random.permutation(batch_size)
        
        # Create the mixup batch
        X_mixup = weights * X + (1 - weights) * X[indices]
        y_mixup = weights.reshape(batch_size, 1) * y + (1 - weights.reshape(batch_size, 1)) * y[indices]
        
        return X_mixup, y_mixup


def load_wav2vec_data(data_dirs, split_ratio=0.8):
    """Load wav2vec features from specified directories."""
    all_data = []
    all_labels = []
    emotion_mapping = {}
    
    print(f"Loading data from directories: {data_dirs}")
    
    for data_dir in data_dirs:
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        print(f"Found {len(npz_files)} files in {data_dir}")
        
        if not npz_files:
            print(f"WARNING: No .npz files found in {data_dir}")
            continue
            
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                features = data["wav2vec_features"]
                
                # Get emotion label - try different possible keys
                if "emotion_class" in data:
                    emotion = data["emotion_class"].item() if hasattr(data["emotion_class"], "item") else data["emotion_class"]
                elif "emotion" in data:
                    emotion_str = str(data["emotion"])
                    if emotion_str not in emotion_mapping:
                        emotion_mapping[emotion_str] = len(emotion_mapping)
                    emotion = emotion_mapping[emotion_str]
                else:
                    # Try to parse from filename
                    filename = os.path.basename(npz_file)
                    for emo in ["angry", "disgust", "fearful", "happy", "neutral", "sad"]:
                        if emo in filename:
                            if emo not in emotion_mapping:
                                emotion_mapping[emo] = len(emotion_mapping)
                            emotion = emotion_mapping[emo]
                            break
                
                all_data.append(features)
                all_labels.append(emotion)
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
    
    if not all_data:
        raise ValueError("No valid data was loaded")
    
    # Convert to arrays
    X = all_data
    y = np.array(all_labels)
    
    # Print data stats
    print(f"Loaded {len(X)} samples with {len(set(y))} emotion classes")
    for emotion, count in zip(*np.unique(y, return_counts=True)):
        print(f"  Class {emotion}: {count} samples")
    
    # Split into train/validation sets
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_point = int(len(indices) * split_ratio)
    
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    # Create training and validation sets
    X_train = [X[i] for i in train_indices]
    y_train = y[train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = y[val_indices]
    
    # Pad sequences for batch processing
    # Find the maximum sequence length
    max_length = max(x.shape[0] for x in X)
    print(f"Maximum sequence length: {max_length}")
    
    # Pad all sequences to the same length
    X_train_padded = np.zeros((len(X_train), max_length, X_train[0].shape[1]))
    X_val_padded = np.zeros((len(X_val), max_length, X_val[0].shape[1]))
    
    for i, x in enumerate(X_train):
        X_train_padded[i, :x.shape[0], :] = x
    
    for i, x in enumerate(X_val):
        X_val_padded[i, :x.shape[0], :] = x
    
    # Convert labels to one-hot encoding
    num_classes = len(set(y))
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes)
    
    return (X_train_padded, y_train_onehot), (X_val_padded, y_val_onehot), num_classes, list(y_train), list(y_val)


def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


def create_attention_crnn_model_v2(input_shape, num_classes):
    """Create an enhanced CNN+RNN+Attention model for wav2vec features."""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = Conv1D(128, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    
    # Bidirectional LSTM for sequence modeling with increased dropout
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
    x = Dropout(0.3)(x)
    
    # Custom attention mechanism with multi-head attention
    query_value_attention_seq = Attention(use_scale=True)([x, x])
    
    # Combine original sequence with attention results
    attention_output = LayerNormalization()(query_value_attention_seq + x)
    
    # Advanced pooling - use attention weighted average instead of simple global average
    context_vector = GlobalAveragePooling1D()(attention_output)
    
    # Enhanced dense layer with BatchNorm and GELU
    x = Dense(256)(context_vector)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Lambda(gelu)(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use Adam with tuned parameters
    optimizer = Adam(learning_rate=1e-3, beta_2=0.98)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save to specified directory
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'attn_crnn_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()


def plot_training_history(history, output_dir):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save to specified directory
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, 'attn_crnn_training_history.png')
    plt.savefig(history_path)
    print(f"Training history saved to {history_path}")
    plt.close()
    
    # Also save history data as numpy array for later analysis
    np.save(os.path.join(output_dir, 'attn_crnn_history.npy'), history.history)


def plot_roc_curves(y_true, y_pred_proba, num_classes, class_names, output_dir):
    """Plot ROC curves for each class."""
    # Binarize the labels for ROC calculation
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend(loc="lower right")
    
    # Save to specified directory
    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, 'attn_crnn_roc_curves.png')
    plt.savefig(roc_path)
    print(f"ROC curves saved to {roc_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train an enhanced Attention-based CRNN model for wav2vec features')
    parser.add_argument('--data_dirs', nargs='+', required=True, help='Directories containing wav2vec feature files')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=6, help='Early stopping patience')
    parser.add_argument('--noise', type=float, default=0.005, help='Gaussian noise sigma for augmentation')
    parser.add_argument('--mixup', type=float, default=0.2, help='MixUp alpha parameter for augmentation')
    parser.add_argument('--disable_augmentation', action='store_true', help='Disable data augmentation')
    args = parser.parse_args()
    
    print("Starting enhanced ATTN-CRNN training with arguments:", args)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("analysis_results", f"attn_crnn_v2_{timestamp}")
    checkpoint_dir = os.path.join("checkpoints", f"attn_crnn_v2_{timestamp}")
    log_dir = os.path.join("logs", f"attn_crnn_v2_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    (X_train, y_train), (X_val, y_val), num_classes, y_train_raw, y_val_raw = load_wav2vec_data(args.data_dirs)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_attention_crnn_model_v2(input_shape, num_classes)
    model.summary()
    
    # Setup callbacks with improved settings
    callbacks = [
        # Save model in native Keras format
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_attn_crnn_model.keras'), 
            save_best_only=True, 
            monitor='val_accuracy'
        ),
        # Early stopping with reasonable patience
        EarlyStopping(
            patience=args.patience, 
            restore_best_weights=True, 
            monitor='val_accuracy'
        ),
        # More aggressive learning rate reduction
        ReduceLROnPlateau(
            factor=0.5, 
            patience=2, 
            min_lr=3e-5, 
            monitor='val_loss',
            verbose=1
        ),
        TensorBoard(log_dir=log_dir)
    ]
    
    # Create data generators
    train_gen = WavDataGenerator(
        X_train, y_train, 
        batch_size=args.batch_size,
        apply_noise=not args.disable_augmentation,
        noise_sigma=args.noise,
        apply_mixup=not args.disable_augmentation,
        mixup_alpha=args.mixup
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    # Generate class names
    class_names = [str(i) for i in range(num_classes)]
    
    # Generate classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, output_dir)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Plot ROC curves
    plot_roc_curves(y_true_classes, y_pred, num_classes, class_names, output_dir)
    
    # Save final model
    model.save(os.path.join(checkpoint_dir, 'final_attn_crnn_model.keras'))
    
    # Create a copy of the best model in the project root for easy access
    shutil.copy(
        os.path.join(checkpoint_dir, 'best_attn_crnn_model.keras'),
        'best_attn_crnn_model.keras'
    )
    
    print("Training complete!")
    print(f"Model saved to {checkpoint_dir}")
    print(f"Analysis results saved to {output_dir}")


if __name__ == "__main__":
    main()
