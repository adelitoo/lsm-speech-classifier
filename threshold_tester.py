import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import time

# ---
# CONFIGURATION
# ---
N_SOGLIE_TO_TEST = [5]
MEL_DATASET_FILE = "mel_spectrogram_dataset.npz"

# ---
# LSM CONSTANTS
# ---
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 2
MEMBRANE_THRESHOLD = 5
SMALL_WORLD_P = 0.2
SMALL_WORLD_K = 100
np.random.seed(42)

# ---
# CLASSIFIER CONSTANTS
# ---
CLASS_NAMES = ["yes", "no", "up", "down", "backward", "bed", "bird", "cat", "dog", "eight", "five", "follow", "forward", "four",
               "go", "happy", "house", "learn", "left", "marvin", "nine", "off", "on", "one", "right", "seven", "sheila", "six",
               "stop", "three", "tree", "two", "visual", "wow", "zero"]

# ---
# MEL-TO-SPIKE FUNCTION
# ---

def convert_mels_to_spikes_temporal(X_mels: np.ndarray, thresholds: list) -> np.ndarray:
    """
    Uses multiple thresholds to encode TIMING information.
    Earlier spikes = higher intensity.
    """
    if not thresholds:
        return np.zeros_like(X_mels, dtype=np.uint8)
    
    # Sort thresholds descending (highest first)
    sorted_thresholds = sorted(thresholds, reverse=True)
    
    # Create output with extra time dimension for multiple spike times
    n_samples, n_mels, n_time = X_mels.shape
    n_threshold_steps = len(sorted_thresholds)
    
    # Expand time axis to accommodate multiple spike times per bin
    X_spikes = np.zeros((n_samples, n_mels, n_time * n_threshold_steps), dtype=np.uint8)
    
    for t_idx, threshold in enumerate(sorted_thresholds):
        # Spikes occur at different time offsets based on threshold
        time_offset = t_idx
        exceeded = X_mels > threshold
        
        # Place spikes in the expanded time dimension
        for time_bin in range(n_time):
            output_time = time_bin * n_threshold_steps + time_offset
            X_spikes[:, :, output_time] = exceeded[:, :, time_bin]
    
    return X_spikes

# ---
# FEATURE EXTRACTION FUNCTION
# ---

def extract_features(lsm, spike_data, features_to_extract, desc=""):
    """
    Processes spike trains through the LSM and extracts a specified set of features.
    """
    all_combined_features = []

    for sample in tqdm(spike_data, desc=desc, leave=False):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        
        feature_dict = lsm.extract_features_from_spikes()
        
        sample_feature_parts = []
        for feature_name in features_to_extract:
            feature_vector = feature_dict[feature_name].copy()
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_vector[feature_vector < 0] = 0
            sample_feature_parts.append(feature_vector)
        
        combined_features = np.concatenate(sample_feature_parts)
        all_combined_features.append(combined_features)

    return np.array(all_combined_features)

def run_lsm_extraction(X_spikes: np.ndarray, y_labels: np.ndarray):
    """
    Takes spike data in memory, runs LSM and returns features.
    """
    num_samples = X_spikes.shape[0]
    num_input_neurons = X_spikes.shape[1]
    # num_time_bins = X_spikes.shape[2] # <-- THIS WAS THE BUGGY LINE
    
    # Calculate w_critico based on this specific dataset's spike density
    # CORRECTED CALCULATION FOR 'I':
    num_time_bins = X_spikes.shape[2]
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    
    beta = SMALL_WORLD_K / 2
    
    if beta == 0:
        w_critico = 0
    else:   
        w_critico = (MEMBRANE_THRESHOLD - 2 * I * REFRACTORY_PERIOD) / beta
        print(f"w_critico: {w_critico}")
    
    # We add a small positive value to weights_mean to ensure 
    # weight_variance (weights_mean * 5) is non-negative, even if w_critico is 0.
    # A max(w_critico, 0.0) would also work, but this is safer.
    weights_mean = max(w_critico, 1e-9) 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    params = SimulationParams(
        num_neurons=NUM_NEURONS, 
        mean_weight=weights_mean, 
        weight_variance=weights_mean * 5,
        num_output_neurons=NUM_OUTPUT_NEURONS, 
        is_random_uniform=False, 
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT, 
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P, 
        small_world_graph_k=SMALL_WORLD_K, 
        input_spike_times=X_train[0]
    )
    lsm = SNN(simulation_params=params)
    
    features_to_use = ["mean_spike_times", "spike_counts"]
    
    X_train_features = extract_features(lsm, X_train, features_to_extract=features_to_use, desc="Extracting train features")
    X_test_features = extract_features(lsm, X_test, features_to_extract=features_to_use, desc="Extracting test features")
    
    return X_train_features, y_train, X_test_features, y_test

# ---
# CLASSIFIER FUNCTION
# ---

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test) -> float:
    """
    Trains classifier and returns accuracy.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42) 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# ---
# MAIN EXPERIMENT WORKFLOW
# ---

def main_experiment():
    """
    Runs the full experiment pipeline for each threshold setting.
    """
    print(f"🚀 Starting experiment... Will test {len(N_SOGLIE_TO_TEST)} configurations.")
    
    # ---
    # STEP 1: Load pre-processed Mel Spectrograms (ONCE)
    # ---
    print(f"Loading mel spectrogram dataset from '{MEL_DATASET_FILE}'...")
    if not Path(MEL_DATASET_FILE).exists():
        print(f"Error: File not found: '{MEL_DATASET_FILE}'")
        print("Please run 'create_mel_dataset.py' first.")
        return
        
    data = np.load(MEL_DATASET_FILE)
    X_mels = data['X_mels']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_mels)} mel spectrograms.")

    # ---
    # STEP 1.5: Pre-filter thresholds
    # ---
    #
    # THIS ENTIRE VALIDATION BLOCK HAS BEEN REMOVED.
    # It's no longer needed because MEMBRANE_THRESHOLD = 5 prevents the
    # w_critico error, so all 100 thresholds are valid.
    #
    
    results = {}
    start_time = time.time()

    # ---
    # STEP 2: Loop through ALL n_soglie values
    # ---
    for idx, n_soglie in enumerate(N_SOGLIE_TO_TEST, 1):
        
        print("\n" + "="*70)
        print(f"STARTING TEST: n_soglie = {n_soglie} ({idx}/{len(N_SOGLIE_TO_TEST)})")
        
        # Calculate thresholds inside the loop
        h = 1.0 / n_soglie
        current_thresholds = [k * h for k in range(1, n_soglie + 1)]
        
        print(f"  -> Using {len(current_thresholds)} thresholds: {[f'{t:.4f}' for t in current_thresholds[:5]]}" + 
              (f" ... {current_thresholds[-1]:.4f}" if len(current_thresholds) > 5 else ""))
    
        
        # 2. Convert Mels to Spikes using temporal encoding
        print("  [Step 1/3] Converting mels to spikes with temporal encoding...")
        X_spikes = convert_mels_to_spikes_temporal(X_mels, current_thresholds)
        
        # Detailed spike statistics
        total_spikes = int(np.sum(X_spikes))
        spike_density = total_spikes / X_spikes.size
        avg_spikes_per_sample = total_spikes / len(X_spikes)
        
        print(f"  -> Output shape: {X_spikes.shape}")
        print(f"  -> Total spikes: {total_spikes:,}")
        print(f"  -> Spike density: {spike_density * 100:.2f}%")
        print(f"  -> Avg spikes per sample: {avg_spikes_per_sample:.1f}")
        
        # 3. Extract LSM Features (This is the slowest part)
        print("  [Step 2/3] Extracting LSM features...")
        X_train_f, y_train_f, X_test_f, y_test_f = run_lsm_extraction(X_spikes, y_labels)
        
        # 4. Train & Evaluate Classifier
        print("  [Step 3/3] Training and evaluating classifier...")
        accuracy = train_and_evaluate_classifier(X_train_f, y_train_f, X_test_f, y_test_f)
        
        # 5. Store and report
        print(f"  -> ✅ Accuracy for n_soglie={n_soglie}: {accuracy*100:.2f}%")
        results[n_soglie] = {
            'accuracy': accuracy,
            'thresholds': current_thresholds,
            'total_spikes': total_spikes,
            'spike_density': spike_density,
            'avg_spikes_per_sample': avg_spikes_per_sample
        }
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n⏱️  Total experiment time: {total_time / 60:.2f} minutes")

    # ---
    # FINAL RESULTS SUMMARY
    # ---
    print("\n" + "🎉"*30)
    print("          EXPERIMENT COMPLETE: FINAL RESULTS")
    print("🎉"*30)
    
    print(f"\nResults for {len(results)} tested configurations:")
    print(f"{'n_soglie':<10} {'Accuracy':<10} {'Total Spikes':<15} {'Spike Density':<15} {'Avg/Sample':<12}")
    print("-" * 70)
    
    best_n = 0
    best_acc = 0

    # Sort results by n_soglie
    sorted_n_soglie = sorted(results.keys())

    for n in sorted_n_soglie:
        result = results[n]
        acc = result['accuracy']
        total = result['total_spikes']
        density = result['spike_density']
        avg = result['avg_spikes_per_sample']
        
        print(f"{n:<10} {acc*100:>6.2f}%    {total:>12,}  {density*100:>10.2f}%     {avg:>10.1f}")
        
        if acc > best_acc:
            best_acc = acc
            best_n = n
            
    print("\n" + "🏆"*30)
    print("                 BEST PERFORMING SETTING")
    print("🏆"*30)
    
    if best_n in results: # Check if any results were produced
        best_result = results[best_n]
        print(f"\nBest performance achieved with n_soglie = {best_n}")
        print(f"  -> Number of thresholds: {len(best_result['thresholds'])}")
        print(f"  -> Threshold range: {best_result['thresholds'][0]:.4f} to {best_result['thresholds'][-1]:.4f}")
        print(f"  -> Total spikes generated: {best_result['total_spikes']:,}")
        print(f"  -> Spike density: {best_result['spike_density']*100:.2f}%")
        print(f"  -> Final Accuracy: {best_acc*100:.2f}%")
    else:
        print("\nNo results were recorded.")

if __name__ == "__main__":
    main_experiment()