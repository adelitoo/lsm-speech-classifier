import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm

SAMPLE_RATE = 16000
DURATION = 1.0
N_MELS = 200
TIME_BINS = 100
SPIKE_THRESHOLDS = [0.60, 0.70, 0.80] 
MAX_SAMPLES_PER_CLASS = 500
VISUALIZE_FIRST_SAMPLE = True 

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
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    if mel_spec_norm.shape[1] != TIME_BINS:
        zoom_factor = TIME_BINS / mel_spec_norm.shape[1]
        mel_spec_norm = zoom(mel_spec_norm, (1, zoom_factor), order=1)
    return mel_spec_norm[:, :TIME_BINS]

def mel_to_spikes(mel_spec: np.ndarray) -> np.ndarray:
    spike_train = np.zeros_like(mel_spec, dtype=np.uint8)
    for threshold in SPIKE_THRESHOLDS:
        spike_train[mel_spec > threshold] = 1
    return spike_train

def visualize_conversion(mel, spikes, filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Conversion for '{filename}'", fontsize=16)
    axes[0].imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Mel Spectrogram")
    axes[0].set_ylabel("Mel Bins")
    axes[1].imshow(spikes, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
    axes[1].set_title("Final Spike Train")
    axes[1].set_ylabel("Mel Bins")
    axes[1].set_xlabel("Time Bins")
    plt.tight_layout()
    plt.show()

def create_dataset():
    """Processes all audio files and saves them into a single .npz file."""
    COMMANDS = ["yes", "no", "up", "down"]
    BASE_DATASET_PATH = Path("speech_commands_v0.02")

    all_spike_trains = []
    all_labels = []
    all_spike_counts = [] 

    print("Starting dataset creation...")

    for label_idx, command in enumerate(COMMANDS):
        print("-" * 50)
        print(f"Processing command: '{command}' (Label: {label_idx})")
        command_dir = BASE_DATASET_PATH / command
        audio_files = sorted(list(command_dir.glob("*.wav")))[:MAX_SAMPLES_PER_CLASS]
        
        command_spike_counts = [] 

        if not audio_files:
            print(f"Warning: No .wav files found for '{command}'. Skipping.")
            continue

        for i, audio_file_path in enumerate(tqdm(audio_files, desc=f"  -> Converting '{command}'")):
            audio_data = load_audio_file(audio_file_path)
            if audio_data is None:
                continue
            mel_spectrogram = audio_to_mel_spectrogram(audio_data)
            spike_train = mel_to_spikes(mel_spectrogram)
            
            num_spikes = np.sum(spike_train)
            command_spike_counts.append(num_spikes)

            all_spike_trains.append(spike_train)
            all_labels.append(label_idx)

            if VISUALIZE_FIRST_SAMPLE and i == 0:
                visualize_conversion(mel_spectrogram, spike_train, audio_file_path.name)
        
        if command_spike_counts:
            avg_spikes_for_command = np.mean(command_spike_counts)
            print(f"  -> DEBUG: Average spikes for '{command}': {avg_spikes_for_command:.2f}")

        all_spike_counts.extend(command_spike_counts) 

    X_spikes = np.array(all_spike_trains, dtype=np.uint8)
    y_labels = np.array(all_labels, dtype=np.int32)

    print("\nDataset creation complete.")
    print(f"Shape of spike train data (X): {X_spikes.shape}")
    print(f"Shape of labels data (y): {y_labels.shape}")

    if all_spike_counts:
        print("\n--- DEBUG: Overall Dataset Statistics ---")
        print(f" > Total samples processed: {len(all_spike_counts)}")
        print(f" > Average spikes per sample (all classes): {np.mean(all_spike_counts):.2f}")
        print(f" > Minimum spikes in a sample: {np.min(all_spike_counts)}")
        print(f" > Maximum spikes in a sample: {np.max(all_spike_counts)}")
        print("----------------------------------------")

    output_filename = "speech_spike_dataset.npz"
    np.savez_compressed(output_filename, X_spikes=X_spikes, y_labels=y_labels)
    print(f"\n✅ Dataset saved to a single file: '{output_filename}'")

if __name__ == "__main__":
    create_dataset()