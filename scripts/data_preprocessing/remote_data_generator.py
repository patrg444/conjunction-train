import numpy as np
import os
import random

# Create directory if it doesn't exist
def generate_data():
    """Generate synthetic emotion data in Wav2Vec format"""
    # Emotion classes
    emotions = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
    
    # Create parent directory
    os.makedirs("/home/ubuntu/emotion_project/data", exist_ok=True)
    
    # Generate 100 samples
    for i in range(100):
        # Random emotion index
        emotion_idx = random.randint(0, len(emotions)-1)
        emotion = emotions[emotion_idx]
        
        # Create feature sequence of random length (50-150 frames)
        seq_length = random.randint(50, 150)
        
        # Wav2vec features are 768-dimensional
        features = np.random.normal(0, 1, (seq_length, 768)).astype(np.float32)
        
        # Save file
        filename = f"/home/ubuntu/emotion_project/crema_d_features/sample_{i:03d}_{emotion}.npz"
        np.savez(filename, 
                wav2vec_features=features,
                emotion=emotion,
                emotion_class=emotion_idx)
        
        print(f"Created {filename}, emotion: {emotion}, length: {seq_length}")
    
    print(f"Generated 100 sample files.")

if __name__ == "__main__":
    generate_data()
