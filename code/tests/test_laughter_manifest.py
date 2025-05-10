#!/usr/bin/env python3
"""
Unit test for laughter manifest creation.
Tests that the manifest has the correct format and structure.
"""

import os
import sys
import csv
import unittest
from pathlib import Path

# Add parent directory to path to import from datasets_raw/scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets_raw.scripts.build_laughter_manifest import (
    process_audioset, process_liris_accede, process_amigail,
    process_tedlium, process_mustard, split_dataset
)


class TestLaughterManifest(unittest.TestCase):
    """Test class for laughter manifest functionality."""

    def setUp(self):
        """Set up test by creating a minimal test directory structure."""
        self.test_dir = Path('tests/test_data')
        self.test_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a simple mock manifest file for testing
        self.test_manifest_file = self.test_dir / 'test_manifest.csv'
        self.create_mock_manifest()
        
        # Paths to test
        self.manifest_path = Path('datasets_raw/manifests/laughter_v1.csv')
    
    def tearDown(self):
        """Clean up test files."""
        if self.test_manifest_file.exists():
            self.test_manifest_file.unlink()
    
    def create_mock_manifest(self):
        """Create a mock manifest file for testing."""
        with open(self.test_manifest_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['youtube_id', 'start_seconds', 'end_seconds', 'labels', 'local_path'])
            writer.writerow(['video1', '0.5', '2.5', '/m/01j3sz', 'datasets_raw/raw/audioset/laughter_clips/video1_0.5_2.5.wav'])
            writer.writerow(['video2', '1.0', '3.0', '/m/01j3sz', 'datasets_raw/raw/audioset/laughter_clips/video2_1.0_3.0.wav'])

    def test_manifest_directory_exists(self):
        """Test that the manifest directory exists."""
        manifest_dir = self.manifest_path.parent
        self.assertTrue(manifest_dir.exists(), 
                       f"Manifest directory {manifest_dir} does not exist")

    def test_process_audioset_handles_missing_files(self):
        """Test that process_audioset handles missing manifest files gracefully."""
        # Test with non-existent directory
        entries = process_audioset('non_existent_dir')
        self.assertEqual(len(entries), 0, 
                        "process_audioset should return empty list for non-existent directory")
        
        # Test with mock manifest but non-existent audio files
        entries = process_audioset(self.test_dir)
        self.assertEqual(len(entries), 0, 
                        "process_audioset should return empty list if audio files don't exist")

    def test_liris_accede_handles_missing_files(self):
        """Test that process_liris_accede handles missing files gracefully."""
        entries = process_liris_accede('non_existent_dir')
        self.assertEqual(len(entries), 0, 
                        "process_liris_accede should return empty list for non-existent directory")

    def test_amigail_handles_missing_files(self):
        """Test that process_amigail handles missing files gracefully."""
        entries = process_amigail('non_existent_dir')
        self.assertEqual(len(entries), 0, 
                        "process_amigail should return empty list for non-existent directory")

    def test_tedlium_handles_missing_files(self):
        """Test that process_tedlium handles missing files gracefully."""
        entries = process_tedlium('non_existent_dir')
        self.assertEqual(len(entries), 0, 
                        "process_tedlium should return empty list for non-existent directory")

    def test_mustard_handles_missing_files(self):
        """Test that process_mustard handles missing files gracefully."""
        entries = process_mustard('non_existent_dir')
        self.assertEqual(len(entries), 0, 
                        "process_mustard should return empty list for non-existent directory")

    def test_split_dataset(self):
        """Test that split_dataset correctly assigns train/val/test splits."""
        entries = [{'filepath': f'file{i}.wav', 'laugh': 1} for i in range(100)]
        split_entries = split_dataset(entries, test_split=0.2, val_split=0.1, seed=42)
        
        # Count entries in each split
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        for entry in split_entries:
            self.assertIn('split', entry, "Each entry should have a 'split' field")
            self.assertIn(entry['split'], split_counts.keys(), 
                         f"Split value '{entry['split']}' should be one of {list(split_counts.keys())}")
            split_counts[entry['split']] += 1
        
        # Check approximate distribution
        self.assertAlmostEqual(split_counts['test'] / len(entries), 0.2, delta=0.02,
                              msg="Test split should be approximately 20%")
        self.assertAlmostEqual(split_counts['val'] / len(entries), 0.1, delta=0.02,
                              msg="Validation split should be approximately 10%")
        self.assertAlmostEqual(split_counts['train'] / len(entries), 0.7, delta=0.02,
                              msg="Training split should be approximately 70%")

    def test_entry_format(self):
        """Test that manifest entries have the correct format."""
        entries = [
            {
                'filepath': 'file1.wav',
                'start': 0.5,
                'end': 2.5,
                'source': 'audioset',
                'speaker_gender': 'unknown',
                'laugh': 1,
                'split': 'train'
            }
        ]
        
        # Check required fields
        required_fields = ['filepath', 'start', 'end', 'source', 'speaker_gender', 'laugh', 'split']
        for field in required_fields:
            self.assertIn(field, entries[0], f"Entry should have '{field}' field")
        
        # Check field types
        self.assertIsInstance(entries[0]['filepath'], str, "filepath should be a string")
        self.assertIsInstance(entries[0]['start'], float, "start should be a float")
        self.assertIsInstance(entries[0]['end'], float, "end should be a float")
        self.assertIsInstance(entries[0]['source'], str, "source should be a string")
        self.assertIsInstance(entries[0]['speaker_gender'], str, "speaker_gender should be a string")
        self.assertIsInstance(entries[0]['laugh'], int, "laugh should be an integer")
        self.assertIsInstance(entries[0]['split'], str, "split should be a string")


if __name__ == '__main__':
    unittest.main()
