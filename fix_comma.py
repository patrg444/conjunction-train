#!/usr/bin/env python3
import sys

# Path to the script on EC2
script_path = "/home/ubuntu/audio_emotion/fixed_v6_script_final.py"

# Read the file
with open(script_path, 'r') as f:
    lines = f.readlines()

# Find the line with the issue (around line 90)
fixed_lines = []
for i, line in enumerate(lines):
    if "tf.keras.backend.set_value(self.model.optimizer.learning_rate" in line and "," not in line:
        # Fix the line by adding the missing comma
        parts = line.strip().split("warmup_lr")
        fixed_line = parts[0] + ", warmup_lr" + parts[1] + "\n"
        fixed_lines.append(fixed_line)
        print(f"Fixed line {i+1}: {line.strip()} -> {fixed_line.strip()}")
    else:
        fixed_lines.append(line)

# Write the fixed content back
with open(script_path, 'w') as f:
    f.writelines(fixed_lines)

print("Fix applied successfully.")
