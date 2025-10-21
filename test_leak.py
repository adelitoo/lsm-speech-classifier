import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Leak coefficient range to test
# Note: leak_coefficient typically ranges from 0 (no leak) to 1 (maximum leak)
LEAK_COEFFICIENTS = np.linspace(0, 1, num=50)  # Test 50 values from 0 to 1

# Base configuration (fixed parameters)
BASE_CONFIG = {
    'num_neurons': 1000,
    'num_output_neurons': 400,
    'refractory_period': 4,
    'membrane_threshold': 2,
    'small_world_p': 0.2,
    'small_world_k': 100
}

# Feature extraction options
USE_BOTH_FEATURES = True  # Use both mean_spike_times and spike_counts

# Smoothing parameters
APPLY_SMOOTHING = True
SMOOTHING_WINDOW = 11  # Smaller window since we have fewer points
SMOOTHING_POLY_ORDER = 3

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    """Load the spike train dataset."""
    print(f"Loading spike train dataset from '{filename}'...")
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    print(f"✅ Loaded {data['X_spikes'].shape[0]} samples")
    return data['X_spikes'], data['y_labels']

def extract_features(lsm, spike_data, use_both_features=True, desc=""):
    """Extract features from LSM."""
    features = []
    for sample in tqdm(spike_data, desc=desc, leave=False):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        feature_dict = lsm.extract_features_from_spikes()
        
        if use_both_features:
            mean_times = feature_dict["mean_spike_times"].copy()
            spike_counts = feature_dict["spike_counts"].copy()
            
            mean_times = np.nan_to_num(mean_times, nan=0.0, posinf=0.0, neginf=0.0)
            spike_counts = np.nan_to_num(spike_counts, nan=0.0, posinf=0.0, neginf=0.0)
            mean_times[mean_times < 0] = 0
            spike_counts[spike_counts < 0] = 0
            
            combined = np.concatenate([mean_times, spike_counts])
            features.append(combined)
        else:
            mean_times = feature_dict["mean_spike_times"].copy()
            mean_times = np.nan_to_num(mean_times, nan=0.0, posinf=0.0, neginf=0.0)
            mean_times[mean_times < 0] = 0
            features.append(mean_times)
    
    return np.array(features)

def calculate_critical_weight(X_spikes, config):
    """Calculate critical weight for the given configuration."""
    num_samples = X_spikes.shape[0]
    num_input_neurons = X_spikes.shape[1]
    num_time_bins = X_spikes.shape[2]
    
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    beta = config['small_world_k'] / 2
    w_critico = (config['membrane_threshold'] - 2 * I * config['refractory_period']) / beta
    return w_critico

def test_leak_coefficient(leak_coef, X_train, X_test, y_train, y_test, X_full, config, use_both_features=True):
    """Test a single leak coefficient value."""
    try:
        weights_mean = calculate_critical_weight(X_full, config)
        
        if weights_mean <= 0 or weights_mean > 10:
            print(f"  ⚠️  Invalid critical weight: {weights_mean:.6f}")
            return None
        
        params = SimulationParams(
            num_neurons=config['num_neurons'],
            mean_weight=weights_mean,
            weight_variance=weights_mean * 5,
            num_output_neurons=config['num_output_neurons'],
            is_random_uniform=False,
            membrane_threshold=config['membrane_threshold'],
            leak_coefficient=leak_coef,
            refractory_period=config['refractory_period'],
            small_world_graph_p=config['small_world_p'],
            small_world_graph_k=config['small_world_k'],
            input_spike_times=X_train[0]
        )
        
        lsm = SNN(simulation_params=params)
        
        X_train_features = extract_features(lsm, X_train, use_both_features, desc="  Train")
        X_test_features = extract_features(lsm, X_test, use_both_features, desc="  Test")
        
        clf = RandomForestClassifier(n_estimators=350, random_state=42, n_jobs=-1)
        clf.fit(X_train_features, y_train)
        
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def plot_leak_coefficient_results(leak_coefs, accuracies, accuracies_smoothed, timestamp, apply_smoothing):
    """Create publication-quality plot for leak coefficient analysis."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    if apply_smoothing:
        # Plot raw data with transparency
        ax.plot(leak_coefs, accuracies, 'o', alpha=0.3, markersize=4, 
                color='lightsteelblue', label="Raw Data")
        # Plot smoothed curve
        ax.plot(leak_coefs, accuracies_smoothed, '-', linewidth=2.5, 
                color='darkblue', label="Smoothed (Savitzky-Golay)")
        
        # Find best on smoothed curve
        best_idx = np.argmax(accuracies_smoothed)
        best_leak = leak_coefs[best_idx]
        best_acc = accuracies_smoothed[best_idx]
    else:
        # Plot raw data only
        ax.plot(leak_coefs, accuracies, 'o-', markersize=6, linewidth=2,
                color='steelblue', label="Test Accuracy")
        
        # Find best on raw data
        best_idx = np.argmax(accuracies)
        best_leak = leak_coefs[best_idx]
        best_acc = accuracies[best_idx]
    
    # Highlight best value
    ax.plot(best_leak, best_acc, 'r*', markersize=20, zorder=5,
            label=f'Best: {best_acc*100:.2f}% at leak={best_leak:.4f}')
    
    # Add reference line at leak=0 (no leak)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.5,
               label='No Leak (leak=0)')
    
    ax.set_title("LSM Classifier Accuracy vs. Leak Coefficient", fontsize=16, fontweight='bold')
    ax.set_xlabel("Leak Coefficient", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if apply_smoothing:
        filename = f"leak_coefficient_analysis_smoothed_{timestamp}.png"
    else:
        filename = f"leak_coefficient_analysis_{timestamp}.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to '{filename}'")
    plt.show()

def main():
    """Main function to test leak coefficient."""
    print("="*70)
    print("LEAK COEFFICIENT ANALYSIS FOR LSM")
    print("="*70)
    
    feature_info = "mean_spike_times + spike_counts" if USE_BOTH_FEATURES else "mean_spike_times only"
    print(f"\nConfiguration:")
    print(f"  Features: {feature_info}")
    print(f"  Leak coefficients to test: {len(LEAK_COEFFICIENTS)} values from {LEAK_COEFFICIENTS[0]:.3f} to {LEAK_COEFFICIENTS[-1]:.3f}")
    print(f"  Smoothing: {'Enabled' if APPLY_SMOOTHING else 'Disabled'}")
    if APPLY_SMOOTHING:
        print(f"  Smoothing window: {SMOOTHING_WINDOW}, poly order: {SMOOTHING_POLY_ORDER}")
    print(f"\nBase configuration:")
    for key, value in BASE_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Load dataset
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    X_full = np.concatenate([X_train, X_test], axis=0)
    print(f"Dataset split: {len(X_train)} train, {len(X_test)} test samples\n")
    
    # Calculate reference critical weight
    w_critico = calculate_critical_weight(X_full, BASE_CONFIG)
    print(f"Reference critical weight (with leak=0): {w_critico:.6f}\n")
    
    # Test each leak coefficient
    print("="*70)
    print("TESTING LEAK COEFFICIENTS")
    print("="*70)
    
    results = []
    leak_coefs_tested = []
    accuracies = []
    
    for leak_coef in tqdm(LEAK_COEFFICIENTS, desc="Overall Progress"):
        print(f"\nTesting leak_coefficient = {leak_coef:.4f}")
        
        accuracy = test_leak_coefficient(
            leak_coef, X_train, X_test, y_train, y_test, X_full, 
            BASE_CONFIG, USE_BOTH_FEATURES
        )
        
        if accuracy is not None:
            print(f"  ✅ Accuracy: {accuracy*100:.2f}%")
            leak_coefs_tested.append(leak_coef)
            accuracies.append(accuracy)
            results.append({
                'leak_coefficient': leak_coef,
                'accuracy': accuracy,
                'critical_weight': w_critico,
                **BASE_CONFIG
            })
        else:
            print(f"  ⚠️  Skipped")
    
    # Process results
    if not results:
        print("\n❌ No valid results obtained!")
        return
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    leak_coefs_array = np.array(leak_coefs_tested)
    accuracies_array = np.array(accuracies)
    
    # Apply smoothing if enabled
    if APPLY_SMOOTHING and len(accuracies_array) >= SMOOTHING_WINDOW:
        print(f"\n🔄 Applying Savitzky-Golay smoothing...")
        accuracies_smoothed = savgol_filter(accuracies_array, SMOOTHING_WINDOW, SMOOTHING_POLY_ORDER)
        print("✅ Smoothing applied")
    else:
        accuracies_smoothed = accuracies_array
        if APPLY_SMOOTHING:
            print(f"\n⚠️  Not enough points for smoothing (need at least {SMOOTHING_WINDOW})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    csv_filename = f"leak_coefficient_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Results saved to '{csv_filename}'")
    
    # Save numpy arrays
    npz_filename = f"leak_coefficient_results_{timestamp}.npz"
    np.savez(npz_filename,
             leak_coefficients=leak_coefs_array,
             accuracies=accuracies_array,
             accuracies_smoothed=accuracies_smoothed,
             critical_weight=w_critico)
    print(f"✅ Arrays saved to '{npz_filename}'")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if APPLY_SMOOTHING and len(accuracies_array) >= SMOOTHING_WINDOW:
        best_idx = np.argmax(accuracies_smoothed)
        best_leak = leak_coefs_array[best_idx]
        best_acc = accuracies_smoothed[best_idx]
        print(f"Best accuracy (smoothed): {best_acc*100:.2f}% at leak_coefficient = {best_leak:.4f}")
    
    best_idx_raw = np.argmax(accuracies_array)
    best_leak_raw = leak_coefs_array[best_idx_raw]
    best_acc_raw = accuracies_array[best_idx_raw]
    print(f"Best accuracy (raw):      {best_acc_raw*100:.2f}% at leak_coefficient = {best_leak_raw:.4f}")
    
    # Accuracy at leak=0 (if tested)
    leak_zero_mask = np.abs(leak_coefs_array) < 0.001
    if np.any(leak_zero_mask):
        acc_at_zero = accuracies_array[leak_zero_mask][0]
        print(f"Accuracy at leak=0:       {acc_at_zero*100:.2f}%")
    
    print(f"\nMean accuracy: {np.mean(accuracies_array)*100:.2f}%")
    print(f"Std accuracy:  {np.std(accuracies_array)*100:.2f}%")
    print(f"Min accuracy:  {np.min(accuracies_array)*100:.2f}%")
    print(f"Max accuracy:  {np.max(accuracies_array)*100:.2f}%")
    
    # Generate plot
    print("\n" + "="*70)
    print("GENERATING PLOT")
    print("="*70)
    plot_leak_coefficient_results(
        leak_coefs_array, accuracies_array, accuracies_smoothed, 
        timestamp, APPLY_SMOOTHING and len(accuracies_array) >= SMOOTHING_WINDOW
    )
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()