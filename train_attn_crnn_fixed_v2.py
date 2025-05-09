#!/usr/bin/env python3
import os
import glob
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator that loads and processes files in batches."""
    
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, max_seq_length=1000):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.file_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Initialize batch arrays with appropriate size
        batch_features = []
        batch_labels = []
        
        # Load and process each file in the batch
        for i in batch_indices:
            try:
                # Load WAV2VEC features
                data = np.load(self.file_paths[i], allow_pickle=True)
                features = data["wav2vec_features"]
                
                # Truncate or pad sequence
                if features.shape[0] > self.max_seq_length:
                    features = features[:self.max_seq_length, :]
                else:
                    # Create padded array with correct shape and data type
                    padded = np.zeros((self.max_seq_length, features.shape[1]), dtype=np.float32)
                    padded[:features.shape[0], :] = features
                    features = padded
                
                batch_features.append(features)
                batch_labels.append(self.labels[i])
            except Exception as e:
                print(f"Error loading file at index {i}: {e}")
        
        # Convert lists to numpy arrays
        X_batch = np.array(batch_features, dtype=np.float32)
        y_batch = to_categorical(np.array(batch_labels), num_classes=len(np.unique(self.labels)))
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_wav2vec_data(data_dirs, split_ratio=0.8, max_seq_length=1000):
    """Load WAV2VEC features from specified directories using memory-efficient approach."""
    all_file_paths = []
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
                # Only load metadata to get the emotion label, not the full features
                data = np.load(npz_file, allow_pickle=True)
                
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

                all_file_paths.append(npz_file)
                all_labels.append(emotion)
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")

    if not all_file_paths:
        raise ValueError("No valid data was loaded")

    # Convert labels to array
    labels = np.array(all_labels)

    # Print data stats
    print(f"Loaded {len(all_file_paths)} samples with {len(set(labels))} emotion classes")
    for emotion, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  Class {emotion}: {count} samples")

    # Split into train/validation sets
    indices = np.arange(len(all_file_paths))
    np.random.shuffle(indices)
    split_point = int(len(indices) * split_ratio)

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # Create training and validation sets
    train_files = [all_file_paths[i] for i in train_indices]
    train_labels = labels[train_indices]
    val_files = [all_file_paths[i] for i in val_indices]
    val_labels = labels[val_indices]

    # Get feature dimension by loading one file
    sample_data = np.load(all_file_paths[0], allow_pickle=True)
    feature_dim = sample_data["wav2vec_features"].shape[1]

    # Create data generators
    train_generator = MemoryEfficientDataGenerator(train_files, train_labels, max_seq_length=max_seq_length)
    val_generator = MemoryEfficientDataGenerator(val_files, val_labels, shuffle=False, max_seq_length=max_seq_length)

    # Return generators and metadata
    num_classes = len(set(labels))
    input_shape = (max_seq_length, feature_dim)
    return train_generator, val_generator, input_shape, num_classes


def create_attention_crnn_model(input_shape, num_classes):
    """Create a CNN+RNN+Attention model for wav2vec features."""
    # Input layer
    inputs = Input(shape=input_shape, name="input_layer")

    # CNN layers for feature extraction
    x = Conv1D(128, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)

    # Bidirectional LSTM for sequence modeling
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Attention mechanism using built-in Attention layer
    attention_layer = tf.keras.layers.Attention()
    x = attention_layer([x, x])
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Classification layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('attn_crnn_confusion_matrix.png')
    print(f"Confusion matrix saved to attn_crnn_confusion_matrix.png")


def plot_training_history(history):
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
    plt.savefig('attn_crnn_training_history.png')
    print(f"Training history saved to attn_crnn_training_history.png")


def main():
    parser = argparse.ArgumentParser(description='Train an Attention-based CRNN model for wav2vec features')
    parser.add_argument('--data_dirs', nargs='+', required=True, help='Directories containing wav2vec feature files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_seq_length', type=int, default=1000, 
                        help='Maximum sequence length (longer sequences will be truncated)')
    args = parser.parse_args()

    print("Starting training with arguments:", args)

    # Load data using memory-efficient generators
    max_seq_length = args.max_seq_length
    train_generator, val_generator, input_shape, num_classes = load_wav2vec_data(
        args.data_dirs, max_seq_length=max_seq_length
    )

    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # Create model
    model = create_attention_crnn_model(input_shape, num_classes)
    model.summary()

    # Setup callbacks
    callbacks = [
        ModelCheckpoint('best_attn_crnn_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=args.patience, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_accuracy'),
        TensorBoard(log_dir=f'logs/attn_crnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    ]

    # Train model using data generators
    # Note: Removed workers and use_multiprocessing parameters
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Evaluate model (using validation generator)
    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation accuracy: {accuracy:.4f}")

    # Get predictions
    val_predictions = []
    val_true_labels = []
    
    for i in range(len(val_generator)):
        X_batch, y_batch = val_generator[i]
        batch_predictions = model.predict(X_batch)
        val_predictions.extend(batch_predictions)
        val_true_labels.extend(y_batch)
    
    val_predictions = np.array(val_predictions)
    val_true_labels = np.array(val_true_labels)
    
    y_pred_classes = np.argmax(val_predictions, axis=1)
    y_true_classes = np.argmax(val_true_labels, axis=1)

    # Generate classification report
    class_names = [str(i) for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)

    # Plot training history
    plot_training_history(history)

    print("Training complete!")


if __name__ == "__main__":
    main()
