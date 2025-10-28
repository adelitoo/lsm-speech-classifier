"""
PURE SIGNAL REDUNDANCY - No jittering, just multiple pathways

This script can be controlled by REDUNDANCY_FACTOR.
- Set to 1: Disables redundancy, creates 80 input neurons.
- Set to >1: Enables redundancy, creates (80 * N) input neurons.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings # Added for cleaner loading

SAMPLE_RATE = 16000 
DURATION = 1.0
N_MELS = 80  # Base number of mel bins
TIME_BINS = 100
SPIKE_THRESHOLDS = [0.15, 0.30, 0.50, 0.75]  
MAX_SAMPLES_PER_CLASS = 1000
VISUALIZE_FIRST_SAMPLE = False

# --- SET THIS TO 1 TO REMOVE REDUNDANCY ---
REDUNDANCY_FACTOR = 1  # Each mel bin repeated 1 time (i.e., no redundancy)
# -------------------------------------------

# --- NEW: Set to True to generate the histogram plot at the end ---
PLOT_THRESHOLD_HISTOGRAM = True
# -------------------------------------------

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
        
    mel_spec_norm = (mel_spec_db - mel_min) / (mel_max - mel_min)
    
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

def convert_mels_to_spikes_hysteresis(mel_spec, thresholds, hysteresis_gap=0.05):
    n_mels, n_time = mel_spec.shape
    n_thresholds = len(thresholds)
    spikes = np.zeros((n_mels, n_time * n_thresholds), dtype=np.uint8)
    
    for t_idx, threshold in enumerate(sorted(thresholds, reverse=True)):
        active = np.zeros(n_mels, dtype=bool)
        lower_bound = threshold - hysteresis_gap
        
        for time_bin in range(n_time):
            rising = (mel_spec[:, time_bin] > threshold) & ~active
            falling = (mel_spec[:, time_bin] < lower_bound) & active
            active[rising] = True
            active[falling] = False
            spikes[:, time_bin * n_thresholds + t_idx] = rising.astype(np.uint8)
            
    return spikes


def create_pure_redundancy(spike_train: np.ndarray, redundancy_factor: int) -> np.ndarray:
    """
    Create pure redundancy by simply repeating each neuron's spike train.
    """
    n_neurons, n_time = spike_train.shape
    redundant = np.repeat(spike_train, redundancy_factor, axis=0)
    return redundant

def visualize_conversion(mel, base_spikes, final_spikes, filename):
    """
    Visualize the conversion.
    Handles both redundant (3 plots) and non-redundant (2 plots) cases.
    """
    
    if REDUNDANCY_FACTOR > 1:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f"Pure Redundancy (Factor {REDUNDANCY_FACTOR}x): '{filename}'", fontsize=16)
        
        # Mel spectrogram
        axes[0].imshow(mel, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"Mel Spectrogram ({N_MELS} bins)")
        axes[0].set_ylabel("Mel Bin")
        
        # Base spike train
        axes[1].imshow(base_spikes, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
        axes[1].set_title(f"Base Spike Train ({np.sum(base_spikes)} spikes)")
        axes[1].set_ylabel(f"Neurons ({N_MELS})")
        
        # Redundant spike train (exact copies)
        axes[2].imshow(final_spikes, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
        axes[2].set_title(f"Redundant Spike Train ({REDUNDANCY_FACTOR}x replication, {np.sum(final_spikes)} total spikes)")
        axes[2].set_ylabel(f"Input Neurons ({N_MELS * REDUNDANCY_FACTOR})")
        axes[2].set_xlabel("Time Bins")
        
        # Add lines showing where replications are
        for i in range(1, REDUNDANCY_FACTOR):
            axes[2].axhline(y=i * N_MELS - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label=f'Copy {i}' if i == 1 else '')
        if REDUNDANCY_FACTOR > 1:
            axes[2].legend()
    
    else: # REDUNDANCY_FACTOR == 1
        fig, axes = plt.subplots(2, 1, figsize=(14, 7))
        fig.suptitle(f"Temporal Encoding (No Redundancy): '{filename}'", fontsize=16)
        
        # Mel spectrogram
        axes[0].imshow(mel, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"Mel Spectrogram ({N_MELS} bins)")
        axes[0].set_ylabel("Mel Bin")
        
        # Base spike train (which is the final spike train)
        axes[1].imshow(base_spikes, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
        axes[1].set_title(f"Spike Train ({np.sum(base_spikes)} spikes)")
        axes[1].set_ylabel(f"Input Neurons ({N_MELS})")
        axes[1].set_xlabel("Time Bins")

    plt.tight_layout()
    plt.show()

# --- NEW: Function to plot the histogram ---
def plot_mel_value_distribution(all_values: list, thresholds: list):
    """
    Plots a histogram of all normalized mel values from the dataset
    and overlays the spike thresholds.
    """
    print("\nGenerating Mel value distribution plot...")
    
    plt.figure(figsize=(12, 7))
    
    # Plot the histogram
    # Using a subset if the dataset is huge, for performance
    if len(all_values) > 10_000_000:
        print("Dataset is large, plotting a random sample of 10M values.")
        plot_values = np.random.choice(all_values, 10_000_000, replace=False)
    else:
        plot_values = all_values
        
    plt.hist(plot_values, bins=100, color='c', edgecolor='k', alpha=0.7, 
             label='Mel Value Distribution (All Samples)')
    
    # Add vertical lines for the thresholds
    colors = ['r', 'g', 'b', 'm']
    for i, thresh in enumerate(thresholds):
        plt.axvline(x=thresh, color=colors[i % len(colors)], linestyle='--', 
                    linewidth=2, label=f'Threshold {thresh:.2f}')

    plt.title('Distribution of Normalized Mel Spectrogram Values vs. Spike Thresholds')
    plt.xlabel('Normalized Mel Value (0.0 to 1.0)')
    plt.ylabel('Frequency (Count)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Use plt.show() to display the plot when the script finishes
    print("...done. Displaying plot.")
    plt.show()

# ---------------------------------------------

def create_dataset():
    """Create dataset, conditionally applying redundancy."""
    COMMANDS = ["yes", "no", "up", "down", "backward", "bed", "bird", "cat", "dog", 
                "eight", "five", "follow"]
    
    BASE_DATASET_PATH = Path("speech_commands_v0.02")
    
    all_spike_trains = []
    all_labels = []
    all_spike_counts = []
    all_mel_values = [] # <-- NEW: To store values for histogram
    
    print("="*60)
    if REDUNDANCY_FACTOR > 1:
        print("CREATING DATASET WITH PURE SIGNAL REDUNDANCY")
        print("="*60)
        print(f"Configuration:")
        print(f"  Redundancy factor: {REDUNDANCY_FACTOR}x")
        print(f"  Input neurons per sample: {N_MELS * REDUNDANCY_FACTOR}")
    else:
        print("CREATING DATASET (Temporal Encoding, NO Redundancy)")
        print("="*60)
        print(f"Configuration:")
        print(f"  Redundancy factor: 1x (Disabled)")
        print(f"  Input neurons per sample: {N_MELS}")

    print(f"  Mel bins: {N_MELS}")
    print(f"  Time bins: {TIME_BINS}")
    print(f"  Thresholds: {SPIKE_THRESHOLDS}")
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
            
            # --- NEW: Collect all mel values ---
            if PLOT_THRESHOLD_HISTOGRAM:
                all_mel_values.extend(mel_spectrogram.flatten())
            # -----------------------------------

            # Convert to base spike train
            base_spike_train = convert_mels_to_spikes_hysteresis(mel_spectrogram, SPIKE_THRESHOLDS)
            
            # --- Conditionally apply redundancy ---
            if REDUNDANCY_FACTOR > 1:
                final_spike_train = create_pure_redundancy(base_spike_train, REDUNDANCY_FACTOR)
                
                # Verify it's exactly redundant (for sanity check)
                if i == 0:
                    for mel_idx in range(min(5, N_MELS)):
                        start = mel_idx * REDUNDANCY_FACTOR
                        end = (mel_idx + 1) * REDUNDANCY_FACTOR
                        group = final_spike_train[start:end, :]
                        if not np.all(group == group[0]):
                             print(f"Warning: Redundancy check failed for mel {mel_idx}")
            else:
                final_spike_train = base_spike_train
            # -------------------------------------
            
            # Track statistics
            num_spikes = np.sum(final_spike_train)
            command_spike_counts.append(num_spikes)
            
            all_spike_trains.append(final_spike_train)
            all_labels.append(label_idx)
            
            # Visualize first sample
            if VISUALIZE_FIRST_SAMPLE and i == 0:
                visualize_conversion(mel_spectrogram, base_spike_train, 
                                   final_spike_train, audio_file.name)
        
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
    
    if REDUNDANCY_FACTOR > 1:
        print(f"  Input neurons: {X_spikes.shape[1]} ({N_MELS} mels × {REDUNDANCY_FACTOR} copies)")
    else:
        print(f"  Input neurons: {X_spikes.shape[1]} ({N_MELS} mels)")
        
    print(f"  Time bins: {X_spikes.shape[2]}")
    print(f"Spike statistics:")
    print(f"  Avg per sample: {np.mean(all_spike_counts):.1f}")
    print(f"  Std: {np.std(all_spike_counts):.1f}")
    print(f"  Min/Max: {np.min(all_spike_counts)} / {np.max(all_spike_counts)}")
    
    if REDUNDANCY_FACTOR > 1:
        base_spikes = np.mean(all_spike_counts) / REDUNDANCY_FACTOR
        print(f"\n  Base spikes (before redundancy): ~{base_spikes:.1f}")
        print(f"  Multiplication factor: {REDUNDANCY_FACTOR}x")
        print(f"  Total spikes (with redundancy): {np.mean(all_spike_counts):.1f}")
    
    print("="*60)
    
    # Save with a new, descriptive name
    if REDUNDANCY_FACTOR > 1:
        output_filename = f"speech_spike_dataset_temporal_redundancy_{REDUNDANCY_FACTOR}x.npz"
    else:
        output_filename = "speech_spike_dataset_temporal_no_redundancy.npz"
        
    np.savez_compressed(output_filename, X_spikes=X_spikes, y_labels=y_labels)
    print(f"\n✅ Saved to '{output_filename}'")
    
    if REDUNDANCY_FACTOR > 1:
        print("\n💡 How this helps:")
        print("   - Each mel frequency is represented by 3 input neurons")
        print("   - All 3 send IDENTICAL spike patterns")
        print("   - But connect to DIFFERENT random subsets of reservoir neurons")
        print("   - LSM integrates across these diverse pathways")
        print("   - Result: More robust features, better generalization")
    else:
        print("\n💡 Redundancy disabled.")
        print("   - Each mel frequency is represented by 1 input neuron.")

    # --- NEW: Call the plotting function at the end ---
    if PLOT_THRESHOLD_HISTOGRAM and all_mel_values:
        plot_mel_value_distribution(all_mel_values, SPIKE_THRESHOLDS)
    # -------------------------------------------------

if __name__ == "__main__":
    create_dataset()