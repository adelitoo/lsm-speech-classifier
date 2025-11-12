import librosa
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings
import argparse
from gammatone import gtgram

SAMPLE_RATE = 16000
DURATION = 1.0
TIME_BINS = 100
SPIKE_THRESHOLDS = [0.70, 0.80, 0.90, 0.95]
HYSTERESIS_GAP = 0.1
MAX_SAMPLES_PER_CLASS = 1000
VISUALIZE_FIRST_SAMPLE = False
REDUNDANCY_FACTOR = 1

np.random.seed(42)


def load_audio_file(filepath: Path) -> np.ndarray | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(
                filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def audio_to_spectrogram(
        audio: np.ndarray,
        n_filters: int,
        filterbank: str) -> np.ndarray:
    if filterbank == 'mel':
        hop_length = max(1, int(len(audio) / TIME_BINS))
        spec = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=n_filters, hop_length=hop_length
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
    else:  # gammatone
        hop_time = len(audio) / (SAMPLE_RATE * TIME_BINS)
        spec = gtgram.gtgram(
            wave=audio,
            fs=SAMPLE_RATE,
            window_time=0.025,
            hop_time=hop_time,
            channels=n_filters,
            f_min=50
        )
        spec_db = 20 * np.log10(spec + 1e-9)
        spec_db = np.maximum(spec_db, spec_db.max() - 80.0)

    spec_min = spec_db.min()
    spec_max = spec_db.max()
    if (spec_max - spec_min) < 1e-8:
        return np.zeros((n_filters, TIME_BINS), dtype=np.float32)

    spec_norm = (spec_db - spec_min) / (spec_max - spec_min + 1e-8)

    if spec_norm.shape[1] != TIME_BINS:
        try:
            zoom_factor = TIME_BINS / spec_norm.shape[1]
            spec_norm = zoom(spec_norm, (1, zoom_factor), order=1)
        except ValueError as e:
            print(
                f"Warning: Zoom failed (shape {spec_norm.shape}, factor {zoom_factor}): {e}")
            return np.zeros((n_filters, TIME_BINS), dtype=np.float32)

    return spec_norm[:, :TIME_BINS]


def convert_spectrogram_to_spikes_hysteresis(
        spectrogram, thresholds, hysteresis_gap=0.05):
    n_filters, n_time = spectrogram.shape
    n_thresholds = len(thresholds)
    spikes = np.zeros((n_filters, n_time * n_thresholds), dtype=np.uint8)

    for t_idx, threshold in enumerate(sorted(thresholds, reverse=True)):
        active = np.zeros(n_filters, dtype=bool)
        lower_bound = threshold - hysteresis_gap
        for time_bin in range(n_time):
            rising = (spectrogram[:, time_bin] > threshold) & ~active
            falling = (spectrogram[:, time_bin] < lower_bound) & active
            active[rising] = True
            active[falling] = False
            output_time = time_bin * n_thresholds + t_idx
            if output_time < spikes.shape[1]:
                spikes[:, output_time] = active.astype(np.uint8)
    return spikes


def create_pure_redundancy(
        spike_train: np.ndarray,
        redundancy_factor: int) -> np.ndarray:
    return np.repeat(spike_train, redundancy_factor, axis=0)


def create_dataset(n_filters: int, filterbank: str):
    COMMANDS = [
        "yes",
        "no",
        "up",
        "visual",
        "backward",
        "stop",
        "bird",
        "cat",
        "nine",
        "eight",
        "zero",
        "follow"]
    BASE_DATASET_PATH = Path("speech_commands_v0.02")

    all_spike_trains = []
    all_labels = []
    all_spike_counts = []

    print(
        f"Creating dataset with filterbank: {filterbank}, filters: {n_filters}")

    for label_idx, command in enumerate(COMMANDS):
        print(f"Processing '{command}'...")
        command_dir = BASE_DATASET_PATH / command
        if not command_dir.is_dir():
            print(f"  Warning: Directory not found, skipping: {command_dir}")
            continue

        audio_files = sorted(list(command_dir.glob("*.wav"))
                             )[:MAX_SAMPLES_PER_CLASS]
        if not audio_files:
            print(f"  Warning: No files found for '{command}'")
            continue

        for audio_file in tqdm(audio_files, desc=f"  Converting"):
            audio_data = load_audio_file(audio_file)
            if audio_data is None:
                continue

            spectrogram = audio_to_spectrogram(
                audio_data, n_filters, filterbank)

            base_spike_train = convert_spectrogram_to_spikes_hysteresis(
                spectrogram,
                SPIKE_THRESHOLDS,
                HYSTERESIS_GAP
            )

            redundant_spike_train = create_pure_redundancy(
                base_spike_train, REDUNDANCY_FACTOR)

            all_spike_trains.append(redundant_spike_train)
            all_labels.append(label_idx)
            all_spike_counts.append(np.sum(redundant_spike_train))

    if not all_spike_trains:
        print("\nERROR: No audio files were successfully processed.")
        return

    X_spikes = np.array(all_spike_trains, dtype=np.uint8)
    y_labels = np.array(all_labels, dtype=np.int32)

    print("\nDataset created successfully.")
    print(f"  Shape: {X_spikes.shape}")
    print(f"  Avg spikes per sample: {np.mean(all_spike_counts):.1f}")

    output_filename = "speech_spike_dataset_pure_redundancy.npz"
    np.savez_compressed(output_filename, X_spikes=X_spikes, y_labels=y_labels)
    print(f"Saved to '{output_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a spike train dataset from audio files.")
    parser.add_argument("--n-filters", type=int, default=128,
                        help="Number of filters for the filterbank.")
    parser.add_argument(
        "--filterbank",
        type=str,
        default="gammatone",
        choices=[
            "mel",
            "gammatone"],
        help="Type of filterbank to use.")

    args = parser.parse_args()

    create_dataset(n_filters=args.n_filters, filterbank=args.filterbank)
