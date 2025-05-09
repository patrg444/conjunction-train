#!/bin/bash
# Shell script to copy the contents of key Python scripts to a single text file for documentation

# Output file path
OUTPUT_FILE="key_scripts_documentation_shell.txt"

# Define key files
KEY_FILES=(
    "scripts/process_ravdess_dataset.py"
    "scripts/multimodal_preprocess.py"
    "scripts/train_branched.py"
    "scripts/analyze_dataset.py"
    "scripts/train_dual_stream.py"
)

# Create output file with header
echo "KEY SCRIPTS DOCUMENTATION (SHELL VERSION)" > "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Loop through each key file
for file_path in "${KEY_FILES[@]}"; do
    if [ -f "$file_path" ]; then
        echo "FILE: $file_path" >> "$OUTPUT_FILE"
        # Create a separator line matching the length of the file path
        printf "=%s\n\n" "$(printf '=%.0s' $(seq 1 $((${#file_path} + 6))))" >> "$OUTPUT_FILE"
        
        # Append file content
        cat "$file_path" >> "$OUTPUT_FILE"
        
        # Add separator for next file
        echo -e "\n\n$(printf '=%.0s' $(seq 1 80))\n\n" >> "$OUTPUT_FILE"
    else
        echo "ERROR: File not found: $file_path" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
done

# Add end marker
echo "END OF DOCUMENTATION" >> "$OUTPUT_FILE"

echo "Documentation created: $OUTPUT_FILE"
