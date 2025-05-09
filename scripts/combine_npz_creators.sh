#!/usr/bin/env bash
#
# Combine the content of the NPZ feature–generation scripts into a single text file.
# Usage: bash scripts/combine_npz_creators.sh
#
OUTFILE="npz_feature_creators.txt"
# List of scripts that generate the .npz feature files
FILES=(
  "scripts/process_all_crema_d.py"
  "scripts/process_all_ravdess.py"
)

# Create or truncate the output file
> "${OUTFILE}"

for file in "${FILES[@]}"; do
  if [[ -f "${file}" ]]; then
    echo "===== ${file} =====" >> "${OUTFILE}"
    cat "${file}" >> "${OUTFILE}"
    echo -e "\n" >> "${OUTFILE}"
  else
    echo "Warning: ${file} not found" >> "${OUTFILE}"
  fi
done

echo "Combined content written to ${OUTFILE}"
