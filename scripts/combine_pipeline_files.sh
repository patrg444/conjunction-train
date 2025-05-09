#!/usr/bin/env bash
#
# Combine the content of core audio-pooling-LSTM pipeline scripts into one text file.
# Usage: bash scripts/combine_pipeline_files.sh
#
OUTFILE="pipeline_files.txt"
# List of pipeline scripts to include
FILES=(
  "scripts/train_audio_pooling_lstm.py"
  "scripts/audio_pooling_generator.py"
  "scripts/facenet_extractor.py"
  "scripts/feature_normalizer.py"
  "scripts/streaming_demo_audio_pooling_lstm.py"
  "scripts/demo_audio_pooling_lstm.py"
  "scripts/compute_embedding_stats.py"
  "scripts/convert_to_stateful.py"
  # Laughter detection pipeline
  "datasets_raw/scripts/fetch_audioset_laughter.sh"
  "datasets_raw/scripts/ingest_liris_accede.py"
  "datasets_raw/scripts/build_laughter_manifest.py"
  "tests/test_laughter_manifest.py"
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

echo "Combined pipeline scripts written to ${OUTFILE}"
