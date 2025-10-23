import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm

SAMPLE_RATE = 16000 # 32hz 8hz, provo con hertz diversi
DURATION = 1.0
N_MELS = 200
TIME_BINS = 100
SPIKE_THRESHOLDS = [0.60, 0.70, 0.80, 0.90] 
MAX_SAMPLES_PER_CLASS = 1000
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

# <-- CHANGED: This is the new, better temporal encoding function
def convert_mels_to_spikes_temporal(mel_spec: np.ndarray, thresholds: list) -> np.ndarray:
    """
    Uses multiple thresholds to encode TIMING information.
    Earlier spikes = higher intensity.
    """
    if not thresholds:
        return np.zeros_like(mel_spec, dtype=np.uint8)
    
    # Sort thresholds descending (highest first)
    sorted_thresholds = sorted(thresholds, reverse=True)
    
    # Create output with extra time dimension for multiple spike times
    n_mels, n_time = mel_spec.shape
    n_threshold_steps = len(sorted_thresholds)
    
    # Expand time axis to accommodate multiple spike times per bin
    X_spikes = np.zeros((n_mels, n_time * n_threshold_steps), dtype=np.uint8)
    
    for t_idx, threshold in enumerate(sorted_thresholds):
        # Spikes occur at different time offsets based on threshold
        time_offset = t_idx
        exceeded = mel_spec > threshold
        
        # Place spikes in the expanded time dimension
        for time_bin in range(n_time):
            output_time = time_bin * n_threshold_steps + time_offset
            X_spikes[:, output_time] = exceeded[:, time_bin]
    
    return X_spikes

def visualize_conversion(mel, spikes, filename):
    # (Visualization function remains the same, though the 'spikes'
    # image will now be 5x wider)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Conversion for '{filename}'", fontsize=16)
    axes[0].imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Mel Spectrogram")
    axes[0].set_ylabel("Mel Bins")
    axes[1].imshow(spikes, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
    axes[1].set_title("Final Spike Train (Temporally Encoded)") # <-- Title changed
    axes[1].set_ylabel("Mel Bins")
    axes[1].set_xlabel("Expanded Time Bins") # <-- Label changed
    plt.tight_layout()
    plt.show()

def create_dataset():
    """Processes all audio files and saves them into a single .npz file."""
    COMMANDS = ["yes", "no", "up", "down", "backward", "bed", "bird", "cat", "dog", "eight", "five", "follow", "forward", "four",
                "go", "happy", "house", "learn", "left", "marvin", "nine", "off", "on", "one", "right", "seven", "sheila", "six",
                "stop", "three", "tree", "two", "visual", "wow", "zero"]
    BASE_DATASET_PATH = Path("speech_commands_v0.02")

    all_spike_trains = []
    all_labels = []
    all_spike_counts = [] 

    print("Starting dataset creation with new temporal encoding...") # <-- Text changed

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
            
            # <-- CHANGED: Call the new function and pass the thresholds
            spike_train = convert_mels_to_spikes_temporal(mel_spectrogram, SPIKE_THRESHOLDS)
            
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
    print(f"Shape of spike train data (X): {X_spikes.shape}") # <-- This shape will now be (35000, 200, 500)
    print(f"Shape of labels data (y): {y_labels.shape}")

    if all_spike_counts:
        print("\n--- DEBUG: Overall Dataset Statistics ---")
        print(f" > Total samples processed: {len(all_spike_counts)}")
        print(f" > Average spikes per sample (all classes): {np.mean(all_spike_counts):.2f}")
        print(f" > Minimum spikes in a sample: {np.min(all_spike_counts)}")
        print(f" > Maximum spikes in a sample: {np.max(all_spike_counts)}")
        print("----------------------------------------")

    output_filename = "speech_spike_dataset.npz" # <-- Changed filename
    np.savez_compressed(output_filename, X_spikes=X_spikes, y_labels=y_labels)
    print(f"\n✅ Dataset saved to a single file: '{output_filename}'")

if __name__ == "__main__":
    create_dataset()