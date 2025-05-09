# ConjunctionTrain: Multimodal Humor and Emotion Recognition Research

This repository contains the code and experiments for research into multimodal humor detection and speech emotion recognition (SER). The project explores various deep learning architectures, feature extraction methods, and fusion techniques to understand and model complex human expression.

## Core Research Areas

This project encompasses two primary research thrusts:

1.  **Multimodal Humor Detection:**
    *   **Objective:** To develop models capable of detecting humor in spoken language by leveraging acoustic, visual, and textual modalities.
    *   **Primary Dataset:** UR-FUNNY dataset.
    *   **Key Models/Techniques:** Explores multimodal fusion models, primarily utilizing transformer-based architectures for text (e.g., XLM-RoBERTa, DeBERTa), various feature extractors for audio (e.g., WavLM, openSMILE), and video features (e.g., OpenFace).
    *   **Key Scripts & Configurations:**
        *   Training: `scripts/train_humor_multimodal_fusion_v2.py`
        *   Model: `models/humor_fusion_model_v2.py`
        *   Dataloader: `dataloaders/humor_multimodal_dataset_v2.py`
        *   Config: `configs/train_humor_multimodal_fusion_v2.yaml`

2.  **Speech Emotion Recognition (SER):**
    *   **Objective:** To build robust models for recognizing six core emotions (happy, sad, angry, neutral, fearful, disgust) from speech.
    *   **Primary Datasets:** RAVDESS, CREMA-D.
    *   **Key Models/Techniques:** Investigation of various architectures including CNNs, LSTMs, Attention mechanisms, and pre-trained models like Wav2Vec2. Features explored include MFCCs, spectrograms, and embeddings from pre-trained audio models.
    *   **Key Scripts & Configurations (Examples - to be refined):**
        *   `scripts/train_wav2vec_lstm.py` (Wav2Vec2 + LSTM approaches)
        *   `scripts/train_spectrogram_cnn_lstm.py` (Spectrogram + CNN-LSTM approaches)
        *   `scripts/train_audio_pooling_lstm.py` (Audio pooling strategies with LSTMs)

## Repository Structure

*   `README.md`: This overview file.
*   `scripts/`: Contains Python scripts for data preprocessing, feature extraction, model training, and evaluation.
    *   *(Planned: `scripts/humor/` - Scripts specific to humor detection.)*
    *   *(Planned: `scripts/emotion/` - Scripts specific to speech emotion recognition.)*
    *   *(Planned: `scripts/utils/` - Common utility functions.)*
*   `models/`: Python modules defining the neural network architectures.
*   `dataloaders/`: Scripts for loading and preparing datasets.
*   `configs/`: YAML configuration files for training experiments and model parameters.
*   `aws-setup/`: Utilities and scripts related to deploying and managing training on AWS EC2 instances. (Note: Many historical scripts will be moved to `archive/`)
*   `docs/`: (Planned) Detailed documentation, supplementary guides, and extended explanations of methodologies.
*   `archive/`: (Planned) Contains older experimental scripts, one-off tests, and historical deployment/monitoring scripts to keep the main directories cleaner.
*   `.gitignore`: Specifies intentionally untracked files (e.g., datasets, model checkpoints, local environment files).
*   `requirements.txt`: Lists Python package dependencies.

## Getting Started

(This section will be updated with instructions on how to set up the environment and run a key part of the project, once the codebase is further organized.)

## Key Technologies

*   Python 3.x
*   PyTorch
*   TensorFlow / Keras
*   PyTorch Lightning (for some experiments)
*   Librosa, OpenCV, Pandas, NumPy, Scikit-learn
*   Hugging Face Transformers (for models like XLM-RoBERTa, DeBERTa, Wav2Vec2)
*   OpenSMILE (for audio feature extraction)
*   OpenFace (for video feature extraction)

---
*This README is a foundational document and will be actively updated as the repository undergoes further organization and cleanup.*
