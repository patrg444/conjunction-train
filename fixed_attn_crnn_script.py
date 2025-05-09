import os
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Attention, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import glob

def load_wav2vec_npz_files(directories, recursive=True):
    """
    Load wav2vec feature npz files from specified directories.
    
    Args:
        directories: List of directories containing npz files
        recursive: Whether to search directories recursively
    
    Returns:
        X: Features array
        y: Labels array
        class_names: List of emotion class names
    """
    print(f"Loading data from directories: {directories}")
    
    features = []
    labels = []
    
    # Emotion mapping based on file naming conventions
    emotion_map = {
        'ANG': 0,  # Angry
        'DIS': 1,  # Disgust
        'FEA': 2,  # Fear
        'HAP': 3,  # Happy
        'SAD': 4,  # Sad
        'NEU': 5,  # Neutral
        'CAL': 5   # Calm (mapped to same as Neutral)
    }

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']
    
    total_files = 0
    
    for directory in directories:
        if recursive:
            # Use a recursive glob to find all npz files
            npz_files = glob.glob(os.path.join(directory, '**', '*.npz'), recursive=True)
        else:
            # Non-recursive search only in the top directory
            npz_files = glob.glob(os.path.join(directory, '*.npz'))
        
        print(f"Found {len(npz_files)} files in {directory}")
        total_files += len(npz_files)
        
        for npz_file in npz_files:
            try:
                filename = os.path.basename(npz_file)
                
                # Try to determine emotion from filename
                emotion_code = None
                if 'cremad_' in filename and '_' in filename:
                    # Format: cremad_XXXX_XXX_EMO_XX.npz
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        emotion_code = parts[3]
                elif 'ravdess_' in filename:
                    # Format: ravdess_XX-XX-XX-XX-XX-XX-XX.npz
                    # where the third number is emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust)
                    parts = filename.replace('.npz', '').split('-')
                    if len(parts) >= 7:
                        emotion_map_ravdess = {'01': 5, '02': 5, '03': 3, '04': 4, '05': 0, '06': 2, '07': 1}
                        emotion_code = emotion_map_ravdess.get(parts[2], None)
                
                if emotion_code is None or (isinstance(emotion_code, str) and emotion_code not in emotion_map):
                    print(f"Skipping {filename} - unknown emotion format")
                    continue
                
                data = np.load(npz_file)
                
                # Different npz files may have different keys
                feature_key = None
                for key in data.keys():
                    if 'feature' in key.lower() or 'embedding' in key.lower() or key == 'wav2vec':
                        feature_key = key
                        break
                
                if feature_key is None:
                    # Try common keys
                    if 'x' in data.keys():
                        feature_key = 'x'
                    elif 'features' in data.keys():
                        feature_key = 'features'
                    elif len(data.keys()) == 1:
                        # If there's only one key, use it
                        feature_key = list(data.keys())[0]
                
                if feature_key is None:
                    print(f"Skipping {filename} - could not find feature data in keys: {data.keys()}")
                    continue
                
                feature_data = data[feature_key]
                
                # Handle different emotion encoding formats
                if isinstance(emotion_code, str):
                    label = emotion_map.get(emotion_code, None)
                else:
                    label = emotion_code
                
                if label is not None:
                    features.append(feature_data)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
    
    if not features:
        raise ValueError("No valid features were loaded. Check your data directories and file formats.")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Loaded {len(features)} samples with {len(class_names)} emotion classes")
    
    # Print class distribution
    for i in range(len(class_names)):
        print(f"  Class {i}: {np.sum(labels == i)} samples")
    
    # Find maximum sequence length
    max_length = max(f.shape[0] for f in features)
    print(f"Maximum sequence length: {max_length}")
    
    # Pad sequences to uniform length
    X = np.zeros((len(features), max_length, features[0].shape[1]))
    for i, f in enumerate(features):
        X[i, :f.shape[0], :] = f
    
    # One-hot encode labels
    y = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
    
    return X, y, class_names

def create_attention_crnn_model(input_shape, num_classes):
    """
    Create an Attention-based CRNN model for emotion classification.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Convolutional blocks
    x = Conv1D(128, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    
    # Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(256)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    
    # Apply attention
    x = tf.keras.layers.Multiply()([x, attention])
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('attn_crnn_confusion_matrix.png')
    plt.close()

def plot_training_history(history):
    """
    Plot and save training history.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('attn_crnn_training_history.png')
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an Attention-based CRNN model for emotion recognition.')
    parser.add_argument('--data_dirs', nargs='+', required=True, help='Directories containing wav2vec feature npz files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--recursive', type=bool, default=True, help='Search directories recursively')
    
    args = parser.parse_args()
    
    print(f"Starting training with arguments: {args}")
    
    # Load data
    X, y, class_names = load_wav2vec_npz_files(args.data_dirs, recursive=args.recursive)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_attention_crnn_model(input_shape, num_classes=len(class_names))
    model.summary()

    # Setup callbacks
    callbacks = [
        ModelCheckpoint('best_attn_crnn_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=args.patience, restore_best_weights=True)
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # Evaluate model
    y_pred = model.predict(X_val)
    
    # Plot results
    plot_confusion_matrix(y_val, y_pred, class_names)
    plot_training_history(history)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        np.argmax(y_val, axis=1),
        np.argmax(y_pred, axis=1),
        target_names=class_names
    ))
    
    print("Training complete. Model saved as 'best_attn_crnn_model.h5'")
    print("Training history saved as 'attn_crnn_training_history.png'")
    print("Confusion matrix saved as 'attn_crnn_confusion_matrix.png'")

if __name__ == "__main__":
    main()
