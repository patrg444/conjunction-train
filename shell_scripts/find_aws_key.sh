#!/usr/bin/env bash
# Script to find AWS SSH key files in the common locations
echo "Searching for AWS SSH key files..."

# Common SSH key locations
LOCATIONS=(
  "$HOME/.ssh"
  "$HOME/Downloads"
  "$HOME/Desktop"
  "$HOME/Documents"
  "$HOME"
)

# Common AWS key patterns
PATTERNS=(
  "aws*.pem"
  "*aws*.pem"
  "*ec2*.pem"
  "*amazon*.pem"
  "id_rsa*"
)

# Search for potential key files
for loc in "${LOCATIONS[@]}"; do
  if [ -d "$loc" ]; then
    echo "Checking $loc..."
    for pattern in "${PATTERNS[@]}"; do
      files=$(find "$loc" -name "$pattern" 2>/dev/null)
      if [ -n "$files" ]; then
        echo "Potential AWS key files found:"
        echo "$files"
      fi
    done
  fi
done

echo "Search complete. If no files were found, please provide the path to your AWS key."
