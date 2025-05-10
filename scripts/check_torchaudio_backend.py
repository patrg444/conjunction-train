import torchaudio
import soundfile

print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Soundfile version: {soundfile.__version__}")

try:
    backend = torchaudio.get_audio_backend()
    print(f"Torchaudio backend: {backend}")
except Exception as e:
    print(f"Error getting torchaudio backend: {e}")

# Try listing available backends if possible (might vary by version)
try:
    available = torchaudio.list_audio_backends()
    print(f"Available backends: {available}")
except AttributeError:
    print("torchaudio.list_audio_backends() not available in this version.")
