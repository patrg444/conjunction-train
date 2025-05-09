# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure for Humor Fusion model training.
- Multi-branch model structure for Smile, Text, and Fusion modeling.
- Automated CelebA manifest preparation for smile classification.

### Changed
- Updated configs/train_humor.yaml to wire correct checkpoint paths.
- Cleaned up obsolete SlowFast test scripts.

### Technical Rationale
#### Why R3D-18?
We keep R3D-18 in the repository as a lightweight, reproducible video baseline. It trains quickly on a single GPU, provides a reasonable reference point for new branches (laughter, smile, text), and plugs cleanly into the fusion head. This makes it ideal for CI forward-pass regression tests and for quantifying gains from the multi-branch architecture.
