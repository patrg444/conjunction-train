#!/usr/bin/env python3
"""
This script generates a synthetic humor dataset for testing the humor detection pipeline.
It creates CSV files in the same format as the UR-FUNNY dataset, allowing the rest of the
pipeline to function without needing the actual dataset.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Sample joke texts with labels (1 = humorous, 0 = not humorous)
JOKES = [
    # Humorous examples (label 1)
    ("I told my wife she was drawing her eyebrows too high. She looked surprised.", 1),
    ("Why don't scientists trust atoms? Because they make up everything.", 1),
    ("I'm on a seafood diet. Every time I see food, I eat it.", 1),
    ("Why don't skeletons fight each other? They don't have the guts.", 1),
    ("What's the best thing about Switzerland? I don't know, but the flag is a big plus.", 1),
    ("I used to play piano by ear, but now I use my hands.", 1),
    ("I was wondering why the ball kept getting bigger and bigger, and then it hit me.", 1),
    ("I'm reading a book about anti-gravity. It's impossible to put down.", 1),
    ("Did you hear about the claustrophobic astronaut? He just needed a little space.", 1),
    ("Why did the scarecrow win an award? He was outstanding in his field.", 1),
    
    # Non-humorous statements (label 0)
    ("The weather forecast predicts rain tomorrow afternoon.", 0),
    ("I need to pick up some groceries on the way home from work.", 0),
    ("The train is scheduled to arrive at 3:45 PM.", 0),
    ("The library closes at 8 PM on weekdays.", 0),
    ("Remember to charge your phone before leaving.", 0),
    ("The meeting has been rescheduled for next Tuesday.", 0),
    ("This book provides an overview of modern physics concepts.", 0),
    ("Please remember to submit your report by Friday.", 0),
    ("The new restaurant opened last week in the downtown area.", 0),
    ("The documentary examines the effects of climate change.", 0),
]

# More examples to expand the dataset
ADDITIONAL_JOKES = [
    # Humorous examples
    ("I told my computer I needed a break, and now it won't stop sending me vacation ads.", 1),
    ("Why did the coffee file a police report? It got mugged.", 1),
    ("What do you call a fake noodle? An impasta.", 1),
    ("How do you organize a space party? You planet.", 1),
    ("Why don't eggs tell jokes? They'd crack each other up.", 1),
    
    # Non-humorous statements
    ("The annual budget meeting will be held in the main conference room.", 0),
    ("Please sign the form and return it to human resources.", 0),
    ("The software update includes several security improvements.", 0),
    ("Make sure to drink plenty of water throughout the day.", 0),
    ("The museum exhibit features artifacts from ancient civilizations.", 0),
]

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("datasets/humor_datasets/synthetic", exist_ok=True)
    os.makedirs("datasets/manifests/humor", exist_ok=True)

def generate_synthetic_dataset(num_samples=200):
    """Generate a synthetic humor dataset with the specified number of samples"""
    # Combine base jokes with additional ones
    all_jokes = JOKES + ADDITIONAL_JOKES
    
    # Create more samples by adding variations
    extended_jokes = []
    
    for text, label in all_jokes:
        extended_jokes.append((text, label))
        
        # Add variations for humorous examples to increase diversity
        if label == 1:
            extended_jokes.append((f"Here's a good one: {text}", 1))
            extended_jokes.append((f"{text} That's hilarious!", 1))
        else:
            extended_jokes.append((f"Note that {text.lower()}", 0))
            extended_jokes.append((f"Just to inform you: {text.lower()}", 0))
    
    # Ensure we have enough samples
    jokes_pool = extended_jokes * (num_samples // len(extended_jokes) + 1)
    
    # Generate random data with the specified split
    np.random.seed(42)  # For reproducibility
    
    # Shuffle the jokes
    np.random.shuffle(jokes_pool)
    
    # Take the required number of samples
    selected_jokes = jokes_pool[:num_samples]
    
    # Create a DataFrame
    df = pd.DataFrame({
        "talk_id": [f"synthetic_{i}" for i in range(len(selected_jokes))],
        "title": ["Synthetic Humor" for _ in range(len(selected_jokes))],
        "text": [joke[0] for joke in selected_jokes],
        "label": [joke[1] for joke in selected_jokes],
        "source": ["synthetic" for _ in range(len(selected_jokes))]
    })
    
    return df

def create_train_val_split(df, train_ratio=0.8):
    """Split the dataset into training and validation sets"""
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split index
    split_idx = int(len(df) * train_ratio)
    
    # Split the data
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    return train_df, val_df

def main():
    """Main function to generate the synthetic humor dataset"""
    setup_directories()
    
    print("Generating synthetic humor dataset...")
    df = generate_synthetic_dataset(num_samples=200)
    
    print("Splitting dataset into train and validation sets...")
    train_df, val_df = create_train_val_split(df)
    
    # Save the datasets
    train_output = "datasets/manifests/humor/ur_funny_train_humor.csv"
    val_output = "datasets/manifests/humor/ur_funny_val_humor.csv"
    
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    
    print(f"\nCreated synthetic training manifest: {train_output}")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Label 0 (non-humorous): {sum(train_df['label'] == 0)}")
    print(f"  Label 1 (humorous): {sum(train_df['label'] == 1)}")
    
    print(f"\nCreated synthetic validation manifest: {val_output}")
    print(f"  Total samples: {len(val_df)}")
    print(f"  Label 0 (non-humorous): {sum(val_df['label'] == 0)}")
    print(f"  Label 1 (humorous): {sum(val_df['label'] == 1)}")
    
    print("\nSynthetic humor dataset generated successfully!")
    print("\nYou can now continue with the humor detection pipeline.")

if __name__ == "__main__":
    main()
