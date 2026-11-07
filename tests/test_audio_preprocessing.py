
from pathlib import Path
import sys

# Add the project root so Python can find our modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_preprocessing import preprocess_audio

# One training recording from your downloaded dataset.
audio_path = (
    Path(r"C:\Users\Ishan\Datasets\ASVspoof2019\LA")
    / "ASVspoof2019_LA_train"
    / "flac"
    / "LA_T_1138215.flac"
)

# Check that the example file exists.
if not audio_path.is_file():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

# Load and preprocess the recording.
audio, sample_rate = preprocess_audio(audio_path)

print("Audio preprocessing successful!")
print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)
print("Duration (seconds):", round(len(audio) / sample_rate, 2))
print("Data type:", audio.dtype)