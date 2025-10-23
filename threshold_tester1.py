import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm

# ---
# CONFIGURATION
# ---
SAMPLE_RATE = 16000
DURATION = 1.0
N_MELS = 200
TIME_BINS = 100
MAX_SAMPLES_PER_CLASS = 1000
VISUALIZE_FIRST_SAMPLE = False 

COMMANDS = ["yes", "no", "up", "down", "backward", "bed", "bird", "cat", "dog", "eight", "five", "follow", "forward", "four",
            "go", "happy", "house", "learn", "left", "marvin", "nine", "off", "on", "one", "right", "seven", "sheila", "six",
            "stop", "three", "tree", "two", "visual", "wow", "zero"]
BASE_DATASET_PATH = Path("speech_commands_v0.02")

OUTPUT_FILENAME = "mel_spectrogram_dataset.npz"

# ---
# FUNCTIONS
# ---

def load_audio_file(filepath: Path) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        return audio
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def audio_to_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    hop_length = int(len(audio) / TIME_BINS)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize from 0 to 1
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    if mel_spec_norm.shape[1] != TIME_BINS:
        zoom_factor = TIME_BINS / mel_spec_norm.shape[1]
        mel_spec_norm = zoom(mel_spec_norm, (1, zoom_factor), order=1)
    return mel_spec_norm[:, :TIME_BINS]

def visualize(mel, filename):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Mel Spectrogram for '{filename}'")
    plt.ylabel("Mel Bins")
    plt.xlabel("Time Bins")
    plt.colorbar(format='%+2.0f dB (normalized)')
    plt.tight_layout()
    plt.show()

def main():
    """Processes all audio files and saves their mel spectrograms."""
    
    all_mels = []
    all_labels = []

    print("Starting dataset creation (Mel Spectrograms)...")
    if not BASE_DATASET_PATH.exists():
        print(f"Error: Dataset directory not found at '{BASE_DATASET_PATH}'")
        print("Please download the speech_commands_v0.02 dataset.")
        return

    for label_idx, command in enumerate(COMMANDS):
        print("-" * 50)
        print(f"Processing command: '{command}' (Label: {label_idx})")
        command_dir = BASE_DATASET_PATH / command
        audio_files = sorted(list(command_dir.glob("*.wav")))[:MAX_SAMPLES_PER_CLASS]
        
        if not audio_files:
            print(f"Warning: No .wav files found for '{command}'. Skipping.")
            continue

        for i, audio_file_path in enumerate(tqdm(audio_files, desc=f"  -> Converting '{command}'")):
            audio_data = load_audio_file(audio_file_path)
            if audio_data is None:
                continue
            
            mel_spectrogram = audio_to_mel_spectrogram(audio_data)
            
            all_mels.append(mel_spectrogram)
            all_labels.append(label_idx)

            if VISUALIZE_FIRST_SAMPLE and i == 0:
                visualize(mel_spectrogram, audio_file_path.name)

    # Use float32 for mels to save space vs float64
    X_mels = np.array(all_mels, dtype=np.float32) 
    y_labels = np.array(all_labels, dtype=np.int32)

    print("\nDataset creation complete.")
    print(f"Shape of mel spectrogram data (X): {X_mels.shape}")
    print(f"Shape of labels data (y): {y_labels.shape}")

    np.savez_compressed(OUTPUT_FILENAME, X_mels=X_mels, y_labels=y_labels)
    print(f"\n✅ Mel spectrogram dataset saved to: '{OUTPUT_FILENAME}'")
    print("You can now run 'run_experiment.py'.")

if __name__ == "__main__":
    main()