from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

def build_augmentations(sample_rate: int):
    """
    Builds the audio augmentation pipeline based on the recipe.
    Args:
        sample_rate (int): The sample rate of the audio.
    Returns:
        audiomentations.Compose: The augmentation pipeline object.
    """
    augment = Compose([
        # Add Gaussian noise with a low amplitude range
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6), 
        # Time stretch audio samples
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
        # Pitch shift audio samples
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3, sample_rate=sample_rate) 
    ])
    return augment

# Example usage:
# if __name__ == '__main__':
#     import numpy as np
#     sample_rate = 16000
#     samples = np.random.uniform(low=-0.5, high=0.5, size=sample_rate * 3).astype(np.float32) # 3 seconds dummy audio
#     
#     augment_pipeline = build_augmentations(sample_rate)
#     
#     print("Original samples shape:", samples.shape)
#     augmented_samples = augment_pipeline(samples=samples, sample_rate=sample_rate)
#     print("Augmented samples shape:", augmented_samples.shape) # Shape might change due to TimeStretch
