import pandas as pd
import torch
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from ser_hubert.data_module import LABEL_MAP # Import the map

# --- Configuration ---
splits_dir = "splits"
train_csv = "train.csv"
output_file = "ser_hubert/class_weights.pt"
label_column = "emotion"
# ---

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load training data
train_path = os.path.join(splits_dir, train_csv)
if not os.path.exists(train_path):
    print(f"Error: Training CSV not found at {train_path}")
    exit(1)

print(f"Loading training data from {train_path}...")
df = pd.read_csv(train_path)

# Preprocess labels: map 'calm' to 'neutral', remove 'surprise'
print("Preprocessing labels (mapping calm->neutral, removing surprise)...")
df[label_column] = df[label_column].replace("calm", "neutral")
df = df[df[label_column] != "surprise"]

# Ensure all labels in the dataframe are in the LABEL_MAP
valid_labels = list(LABEL_MAP.keys())
df = df[df[label_column].isin(valid_labels)]
print(f"Filtered dataframe to {len(df)} rows with labels in LABEL_MAP.")

if df.empty:
    print("Error: No valid training data found after filtering.")
    exit(1)

# Get labels for weight calculation
labels = df[label_column].tolist()
unique_labels = sorted(list(set(labels)))
print(f"Unique labels found for weight calculation: {unique_labels}")

# Check if unique labels match LABEL_MAP keys exactly
if set(unique_labels) != set(LABEL_MAP.keys()):
    print("Warning: Unique labels in CSV do not exactly match LABEL_MAP keys!")
    print(f"  CSV unique: {unique_labels}")
    print(f"  MAP keys:   {sorted(list(LABEL_MAP.keys()))}")
    # Decide how to handle this - error out or proceed with caution?
    # For now, proceed but use labels present in the data for calculation.

# Map string labels to integer IDs using LABEL_MAP for sklearn function
label_ids = [LABEL_MAP[lbl] for lbl in labels]
unique_label_ids = sorted(list(set(label_ids)))

print(f"Calculating class weights for labels: {unique_labels} (IDs: {unique_label_ids})")

# Calculate class weights using sklearn
# 'balanced' mode automatically computes weights inversely proportional to class frequencies
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.array(unique_label_ids), # Pass the unique integer IDs found
    y=np.array(label_ids)              # Pass all integer IDs
)

# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

print(f"Calculated weights: {class_weights_tensor}")

# Save the tensor
print(f"Saving weights to {output_file}...")
torch.save(class_weights_tensor, output_file)
print("Class weights saved successfully.")
