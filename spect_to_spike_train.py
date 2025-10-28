import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings # <-- Added import

SAMPLE_RATE = 16000 # 8Hz / 32Hz
DURATION = 1.0
N_MELS = 200  # Base number of mel bins
TIME_BINS = 100
SPIKE_THRESHOLDS = [0.60, 0.70, 0.80, 0.90]     
HYSTERESIS_GAP = 0.05 # <-- Added Hysteresis Gap
MAX_SAMPLES_PER_CLASS = 1000
VISUALIZE_FIRST_SAMPLE = False

# PURE REDUNDANCY - just replication
REDUNDANCY_FACTOR = 1  

np.random.seed(42)

def load_audio_file(filepath: Path) -> np.ndarray | None:
    """Load audio file"""
    try:
        # Suppress warnings if file is < DURATION
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
            
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def audio_to_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mel spectrogram"""
    # Ensure hop_length is at least 1
    hop_length = max(1, int(len(audio) / TIME_BINS))
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Handle silence or near-silence
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    if (mel_max - mel_min) < 1e-8:
        return np.zeros((N_MELS, TIME_BINS), dtype=np.float32) # Return zeros for silence
        
    mel_spec_norm = (mel_spec_db - mel_min) / (mel_max - mel_min + 1e-8)
    
    # Resize to exactly TIME_BINS
    if mel_spec_norm.shape[1] != TIME_BINS:
        try:
            zoom_factor = TIME_BINS / mel_spec_norm.shape[1]
            mel_spec_norm = zoom(mel_spec_norm, (1, zoom_factor), order=1)
        except ValueError as e:
            print(f"Warning: Zoom failed (shape {mel_spec_norm.shape}, factor {zoom_factor}): {e}")
            # Fallback: create an empty array of the correct shape
            return np.zeros((N_MELS, TIME_BINS), dtype=np.float32)
            
    # Ensure shape is exact after zoom/padding
    return mel_spec_norm[:, :TIME_BINS]

# --- This is the old function (which is no longer called) ---
def convert_mels_to_spikes_temporal(mel_spec: np.ndarray, thresholds: list) -> np.ndarray:
    """Convert mel spectrogram to temporal spike encoding"""
    if not thresholds:
        return np.zeros_like(mel_spec, dtype=np.uint8)
    
    sorted_thresholds = sorted(thresholds, reverse=True)
    n_mels, n_time = mel_spec.shape
    n_threshold_steps = len(sorted_thresholds)
    
    X_spikes = np.zeros((n_mels, n_time * n_threshold_steps), dtype=np.uint8)
    
    for t_idx, threshold in enumerate(sorted_thresholds):
        time_offset = t_idx
        exceeded = mel_spec > threshold
        
        for time_bin in range(n_time):
            output_time = time_bin * n_threshold_steps + time_offset
            if output_time < X_spikes.shape[1]: # Bounds check
                X_spikes[:, output_time] = exceeded[:, time_bin]
    
    return X_spikes

# --- HERE IS THE RESTORED HYSTERESIS FUNCTION ---
def convert_mels_to_spikes_hysteresis(mel_spec, thresholds, hysteresis_gap=0.05):
    """
    Converts mel spectrogram to spikes using hysteresis.
    Outputs a '1' for every time step the neuron is in the 'active' state.
    """
    n_mels, n_time = mel_spec.shape
    n_thresholds = len(thresholds)
    
    # Output shape is (n_mels, n_time * n_thresholds)
    # This interleaves the threshold channels
    spikes = np.zeros((n_mels, n_time * n_thresholds), dtype=np.uint8)
    
    # We iterate through thresholds from high to low
    for t_idx, threshold in enumerate(sorted(thresholds, reverse=True)):
        active = np.zeros(n_mels, dtype=bool) # Memory for each neuron
        lower_bound = threshold - hysteresis_gap
        
        for time_bin in range(n_time):
            # Conditions to change state
            rising = (mel_spec[:, time_bin] > threshold) & ~active
            falling = (mel_spec[:, time_bin] < lower_bound) & active
            
            # Update the state
            active[rising] = True
            active[falling] = False
            
            # --- Sustained Pulse Logic ---
            # Output the 'active' state for this time step
            # The t_idx ensures we write to the correct channel
            output_time = time_bin * n_thresholds + t_idx
            if output_time < spikes.shape[1]: # Bounds check
                spikes[:, output_time] = active.astype(np.uint8)
            
    return spikes
# ----------------------------------------------------

def create_pure_redundancy(spike_train: np.ndarray, redundancy_factor: int) -> np.ndarray:
    """
    Create pure redundancy by simply repeating each neuron's spike train.
    
    NO noise, NO jitter, NO modifications - just exact copies.
    
    The benefit comes from:
    - Each copy connects to DIFFERENT reservoir neurons (random connectivity)
    - LSM can integrate across multiple pathways
    - More robust to individual connection failures
    
    Args:
        spike_train: Original spike train (n_neurons, n_time)
        redundancy_factor: How many times to repeat each neuron
    
    Returns:
        Redundant spike train (n_neurons * redundancy_factor, n_time)
    """
    n_neurons, n_time = spike_train.shape
    
    # Simply repeat each row (neuron) redundancy_factor times
    # np.repeat does this efficiently
    redundant = np.repeat(spike_train, redundancy_factor, axis=0)
    
    return redundant

def visualize_conversion(mel, base_spikes, redundant_spikes, filename):
    """Visualize the conversion with pure redundancy"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Pure Redundancy (No Jitter): '{filename}'", fontsize=16)
    
    # Mel spectrogram
    axes[0].imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f"Mel Spectrogram ({N_MELS} bins)")
    axes[0].set_ylabel("Mel Bin")
    
    # Base spike train
    axes[1].imshow(base_spikes, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
    axes[1].set_title(f"Base Spike Train ({np.sum(base_spikes)} spikes)")
    axes[1].set_ylabel(f"Neurons ({N_MELS})")
    
    # Redundant spike train (exact copies)
    axes[2].imshow(redundant_spikes, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
    axes[2].set_title(f"Redundant Spike Train ({REDUNDANCY_FACTOR}x replication, {np.sum(redundant_spikes)} total spikes)")
    axes[2].set_ylabel(f"Input Neurons ({N_MELS * REDUNDANCY_FACTOR})")
    axes[2].set_xlabel("Time Bins")
    
    # Add lines showing where replications are
    for i in range(1, REDUNDANCY_FACTOR):
        axes[2].axhline(y=i * N_MELS - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label=f'Copy {i}' if i == 1 else '')
    
    if REDUNDANCY_FACTOR > 1:
        axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def create_dataset():
    """Create dataset with pure signal redundancy (no jitter)"""
    COMMANDS = ["yes", "no", "up", "down", "backward", "bed", "bird", "cat", "dog", 
                "eight", "five", "follow"]
    
    BASE_DATASET_PATH = Path("speech_commands_v0.02")
    
    all_spike_trains = []
    all_labels = []
    all_spike_counts = []
    
    print("="*60)
    print("CREATING DATASET WITH PURE SIGNAL REDUNDANCY")
    print("="*60)
    print(f"Configuration:")
    print(f"  Mel bins: {N_MELS}")
    print(f"  Time bins: {TIME_BINS}")
    print(f"  Redundancy factor: {REDUNDANCY_FACTOR}x")
    print(f"  Input neurons per sample: {N_MELS * REDUNDANCY_FACTOR}")
    print(f"  Thresholds: {SPIKE_THRESHOLDS}")
    print(f"  Encoding: Hysteresis (Gap: {HYSTERESIS_GAP})") # <-- Updated print
    print(f"\n  Strategy: Each spike train sent to {REDUNDANCY_FACTOR} different")
    print(f"            input neurons with IDENTICAL patterns.")
    print(f"            Benefit from DIFFERENT reservoir connections only.")
    print("="*60 + "\n")
    
    for label_idx, command in enumerate(COMMANDS):
        print(f"Processing '{command}' (label {label_idx})...")
        command_dir = BASE_DATASET_PATH / command
        
        # Check if directory exists
        if not command_dir.is_dir():
            print(f"  Warning: Directory not found, skipping: {command_dir}")
            continue

        audio_files = sorted(list(command_dir.glob("*.wav")))[:MAX_SAMPLES_PER_CLASS]
        
        if not audio_files:
            print(f"  Warning: No files found for '{command}'")
            continue
        
        command_spike_counts = []
        
        for i, audio_file in enumerate(tqdm(audio_files, desc=f"  Converting")):
            # Load and convert audio
            audio_data = load_audio_file(audio_file)
            if audio_data is None:
                continue
            
            mel_spectrogram = audio_to_mel_spectrogram(audio_data)
            
            # --- MODIFIED: Calling hysteresis function ---
            base_spike_train = convert_mels_to_spikes_hysteresis(
                mel_spectrogram, 
                SPIKE_THRESHOLDS, 
                HYSTERESIS_GAP
            )
            # ---------------------------------------------
            
            # Create PURE redundancy - just exact replication
            redundant_spike_train = create_pure_redundancy(base_spike_train, REDUNDANCY_FACTOR)
            
            # Verify it's exactly redundant (for sanity check)
            if i == 0 and REDUNDANCY_FACTOR > 1: # Added check for factor > 1
                # Check first sample: each group of REDUNDANCY_FACTOR rows should be identical
                for mel_idx in range(min(5, N_MELS)):  # Check first 5 mel bins
                    start = mel_idx * REDUNDANCY_FACTOR
                    end = (mel_idx + 1) * REDUNDANCY_FACTOR
                    group = redundant_spike_train[start:end, :]
                    # All rows in group should be identical
                    assert np.all(group == group[0]), f"Redundancy check failed for mel {mel_idx}"
            
            # Track statistics
            num_spikes = np.sum(redundant_spike_train)
            command_spike_counts.append(num_spikes)
            
            all_spike_trains.append(redundant_spike_train)
            all_labels.append(label_idx)
            
            # Visualize first sample
            if VISUALIZE_FIRST_SAMPLE and i == 0:
                visualize_conversion(mel_spectrogram, base_spike_train, 
                                   redundant_spike_train, audio_file.name)
        
        if command_spike_counts:
            avg_spikes = np.mean(command_spike_counts)
            print(f"  → Avg spikes: {avg_spikes:.1f}")
        
        all_spike_counts.extend(command_spike_counts)
    
    # --- Check if any data was processed ---
    if not all_spike_trains:
        print("\n" + "="*60)
        print("ERROR: No audio files were successfully processed.")
        print(f"Please check the path: {BASE_DATASET_PATH.resolve()}")
        print("="*60)
        return # Exit if no data
    # ---------------------------------------

    # Convert to arrays
    X_spikes = np.array(all_spike_trains, dtype=np.uint8)
    y_labels = np.array(all_labels, dtype=np.int32)
    
    # Statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(X_spikes)}")
    print(f"Shape: {X_spikes.shape}")
    print(f"  Input neurons: {X_spikes.shape[1]} ({N_MELS} mels × {REDUNDANCY_FACTOR} copies)")
    print(f"  Time bins: {X_spikes.shape[2]}")
    print(f"Spike statistics:")
    print(f"  Avg per sample: {np.mean(all_spike_counts):.1f}")
    print(f"  Std: {np.std(all_spike_counts):.1f}")
    print(f"  Min/Max: {np.min(all_spike_counts)} / {np.max(all_spike_counts)}")
    
    # Note about redundancy
    if REDUNDANCY_FACTOR > 1: # Only print this if redundancy is active
        base_spikes = np.mean(all_spike_counts) / REDUNDANCY_FACTOR
        print(f"\n  Base spikes (before redundancy): ~{base_spikes:.1f}")
        print(f"  Multiplication factor: {REDUNDANCY_FACTOR}x")
        print(f"  Total spikes (with redundancy): {np.mean(all_spike_counts):.1f}")
    
    print("="*60)
    
    # Save
    output_filename = "speech_spike_dataset_pure_redundancy.npz"
    np.savez_compressed(output_filename, X_spikes=X_spikes, y_labels=y_labels)
    print(f"\n✅ Saved to '{output_filename}'")
    
    if REDUNDANCY_FACTOR > 1: # Only print if redundancy is active
        print("\n💡 How this helps:")
        print("   - Each mel frequency is represented by 3 input neurons")
        print("   - All 3 send IDENTICAL spike patterns")
        print("   - But connect to DIFFERENT random subsets of reservoir neurons")
        print("   - LSM integrates across these diverse pathways")
        print("   - Result: More robust features, better generalization")
    else:
        print("\n💡 Redundancy disabled.")

if __name__ == "__main__":
    create_dataset()