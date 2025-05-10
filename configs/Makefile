# Makefile for Emotion Recognition with Laughter Detection
# This file provides commands to streamline the data acquisition and processing
# for the laughter/humor detection components of the emotion recognition pipeline.

.PHONY: all clean setup test laughter_data audioset liris_accede amigail tedlium mustard unified_manifest

# Default target: show help
all: help

# Setup directories
setup:
	@echo "Creating directory structure..."
	@mkdir -p datasets/raw/audioset
	@mkdir -p datasets/raw/liris-accede
	@mkdir -p datasets/raw/amigail
	@mkdir -p datasets/raw/ted-lium
	@mkdir -p datasets/raw/mustard
	@mkdir -p datasets/manifests
	@mkdir -p tests/test_data

# Download and process each dataset
laughter_data: setup audioset liris_accede amigail tedlium mustard unified_manifest

# Process AudioSet laughter dataset
audioset:
	@echo "Processing AudioSet laughter dataset..."
	@bash datasets/scripts/fetch_audioset_laughter.sh

# Process LIRIS-ACCEDE humor dataset
liris_accede:
	@echo "Processing LIRIS-ACCEDE humor dataset..."
	@python datasets/scripts/ingest_liris_accede.py

# Process AMIGAIL corpus
amigail:
	@echo "Processing AMIGAIL corpus..."
	@echo "Note: You need to manually download AMIGAIL corpus to datasets/raw/amigail"
	@echo "      and prepare laughter_annotations.csv file."

# Process TED-LIUM dataset with laughter annotations
tedlium:
	@echo "Processing TED-LIUM dataset with laughter annotations..."
	@echo "Note: You need to manually download TED-LIUM corpus to datasets/raw/ted-lium"
	@echo "      and prepare laughter_segments.csv file."

# Process MUStARD dataset
mustard:
	@echo "Processing MUStARD dataset..."
	@echo "Note: You need to manually download MUStARD corpus to datasets/raw/mustard"
	@echo "      and prepare humor_annotations.csv file."

# Build unified manifest
unified_manifest:
	@echo "Building unified laughter manifest..."
	@python datasets/scripts/build_laughter_manifest.py --add_negative

# Run tests
test:
	@echo "Running tests..."
	@python -m unittest tests/test_laughter_manifest.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f datasets/manifests/*.csv

# Show help
help:
	@echo "Emotion Recognition with Laughter Detection"
	@echo "----------------------------------------"
	@echo "Available targets:"
	@echo "  setup            - Create directory structure"
	@echo "  laughter_data    - Process all laughter datasets"
	@echo "  audioset         - Process AudioSet laughter dataset"
	@echo "  liris_accede     - Process LIRIS-ACCEDE humor dataset"
	@echo "  amigail          - Process AMIGAIL corpus"
	@echo "  tedlium          - Process TED-LIUM dataset"
	@echo "  mustard          - Process MUStARD dataset"
	@echo "  unified_manifest - Build unified laughter manifest"
	@echo "  test             - Run tests"
	@echo "  clean            - Clean generated files"
	@echo "  help             - Show this help"
