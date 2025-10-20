import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from datetime import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Parameter ranges to sample from
PARAM_RANGES = {
    'num_neurons': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 
                    2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000],
    'num_output_neurons': [200, 300, 400, 500, 600, 700, 800, 900],
    'refractory_period': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    'membrane_threshold': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75,
                           6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10],
    'small_world_p': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    'small_world_k': [100, 150, 200, 250, 300, 350, 400, 500],
}

# Base configuration
BASE_CONFIG = {
    'num_neurons': 1000,
    'num_output_neurons': 400,
    'leak_coefficient': 0,
    'refractory_period': 4,
    'membrane_threshold': 2,
    'small_world_p': 0.2,
    'small_world_k': 100
}

# Feature extraction options
USE_BOTH_FEATURES = True  # Set to True to use both mean_spike_times and spike_counts

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    """Load the spike train dataset."""
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    return data['X_spikes'], data['y_labels']

def extract_features(lsm, spike_data, use_both_features=True, desc=""):
    """Extract features from LSM - can use mean_spike_times, spike_counts, or both."""
    features = []
    for sample in tqdm(spike_data, desc=desc, leave=False):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        feature_dict = lsm.extract_features_from_spikes()
        
        if use_both_features:
            # Use BOTH features (more informative)
            mean_times = feature_dict["mean_spike_times"].copy()
            spike_counts = feature_dict["spike_counts"].copy()
            
            # Clean features
            mean_times[mean_times < 0] = 0
            spike_counts[spike_counts < 0] = 0
            
            # Concatenate both features
            combined = np.concatenate([mean_times, spike_counts])
            features.append(combined)
        else:
            # Use only mean_spike_times (faster)
            mean_times = feature_dict["mean_spike_times"]
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
    return w_critico, I, beta

def is_valid_configuration(config, I):
    """Check if a configuration will produce valid weights."""
    beta = config['small_world_k'] / 2
    w_critico = (config['membrane_threshold'] - 2 * I * config['refractory_period']) / beta
    
    if w_critico <= 0:
        return False, f"Critical weight would be {w_critico:.6f} (must be > 0)"
    
    variance = w_critico * 5
    if variance <= 0:
        return False, f"Variance would be {variance:.6f} (must be > 0)"
    
    return True, "Valid"

def test_configuration(config, X_train, X_test, y_train, y_test, X_full, use_both_features=True):
    """Test a single parameter configuration and return accuracy."""
    try:
        weights_mean, I, beta = calculate_critical_weight(X_full, config)
        
        is_valid, message = is_valid_configuration(config, I)
        if not is_valid:
            print(f"  ⚠️  Skipping invalid config: {message}")
            return None, None
        
        if weights_mean < 0.001 or weights_mean > 10:
            print(f"  ⚠️  Skipping: Critical weight {weights_mean:.6f} is out of reasonable range")
            return None, None
        
        params = SimulationParams(
            num_neurons=config['num_neurons'],
            mean_weight=weights_mean,
            weight_variance=weights_mean * 5,
            num_output_neurons=config['num_output_neurons'],
            is_random_uniform=False,
            membrane_threshold=config['membrane_threshold'],
            leak_coefficient=config['leak_coefficient'],
            refractory_period=config['refractory_period'],
            small_world_graph_p=config['small_world_p'],
            small_world_graph_k=config['small_world_k'],
            input_spike_times=X_train[0]
        )
        
        lsm = SNN(simulation_params=params)
        
        X_train_features = extract_features(lsm, X_train, use_both_features, desc="Train")
        X_test_features = extract_features(lsm, X_test, use_both_features, desc="Test")
        
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_train_features, y_train)
        
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, weights_mean
    
    except Exception as e:
        print(f"  ❌ Error with config: {e}")
        return None, None

def plot_single_param_results(df, param_name, timestamp):
    """Create line plot for single parameter sweep."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['value'], df['accuracy'], 'o-', linewidth=2, markersize=8)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Effect of {param_name} on Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(1, 2, 2)
    plt.plot(df['value'], df['critical_weight'], 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Critical Weight', fontsize=12)
    plt.title(f'Critical Weight vs {param_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'single_param_{param_name}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Plot saved to '{filename}'")
    plt.close()

def plot_all_params_heatmap(all_results_df, timestamp):
    """Create heatmap showing all single-parameter sweep results."""
    # Get unique parameters
    params = all_results_df['parameter'].unique()
    
    if len(params) < 2:
        print("  ⚠️  Need at least 2 parameters for heatmap")
        return
    
    # Create figure with subplots
    n_params = len(params)
    fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
    
    if n_params == 1:
        axes = [axes]
    
    for idx, param in enumerate(params):
        param_data = all_results_df[all_results_df['parameter'] == param]
        
        # Create a matrix for heatmap (single column)
        values = param_data['value'].values
        accuracies = param_data['accuracy'].values
        
        # Reshape for heatmap
        matrix = accuracies.reshape(-1, 1)
        
        sns.heatmap(
            matrix,
            ax=axes[idx],
            cmap='RdYlGn',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Accuracy'},
            yticklabels=[f'{v:.2f}' for v in values],
            xticklabels=[''],
            vmin=0,
            vmax=1
        )
        axes[idx].set_title(f'{param}', fontsize=12)
        axes[idx].set_ylabel('Parameter Value', fontsize=10)
    
    plt.tight_layout()
    filename = f'all_params_heatmap_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Heatmap saved to '{filename}'")
    plt.close()

def plot_param_comparison(all_results_df, timestamp):
    """Create comparison plot showing all parameters together."""
    params = all_results_df['parameter'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, param in enumerate(params):
        if idx >= len(axes):
            break
            
        param_data = all_results_df[all_results_df['parameter'] == param].sort_values('value')
        
        axes[idx].plot(param_data['value'], param_data['accuracy'], 'o-', linewidth=2, markersize=8)
        axes[idx].set_xlabel(param, fontsize=11)
        axes[idx].set_ylabel('Accuracy', fontsize=11)
        axes[idx].set_title(f'Effect of {param}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])
        
        # Highlight best value
        best_idx = param_data['accuracy'].idxmax()
        best_row = param_data.loc[best_idx]
        axes[idx].scatter([best_row['value']], [best_row['accuracy']], 
                         color='red', s=200, marker='*', zorder=5, 
                         label=f'Best: {best_row["accuracy"]*100:.1f}%')
        axes[idx].legend()
    
    # Hide unused subplots
    for idx in range(len(params), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    filename = f'param_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Comparison plot saved to '{filename}'")
    plt.close()

def run_single_param_sweep(param_name, use_both_features=True):
    """Test one parameter at a time."""
    print("="*70)
    print(f"TESTING PARAMETER: {param_name}")
    print("="*70)
    
    feature_info = "mean_spike_times + spike_counts" if use_both_features else "mean_spike_times only"
    print(f"Using features: {feature_info}\n")
    
    print("Loading dataset...")
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    X_full = np.concatenate([X_train, X_test], axis=0)
    print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples\n")
    
    results = []
    
    if param_name not in PARAM_RANGES:
        print(f"Error: {param_name} not in parameter ranges")
        return None
    
    for value in PARAM_RANGES[param_name]:
        config = BASE_CONFIG.copy()
        config[param_name] = value
        
        print(f"\n{param_name} = {value}")
        accuracy, w_crit = test_configuration(config, X_train, X_test, y_train, y_test, X_full, use_both_features)
        
        if accuracy is not None:
            print(f"  ✅ Accuracy: {accuracy*100:.2f}% | Critical weight: {w_crit:.6f}")
            results.append({
                'parameter': param_name,
                'value': value,
                'accuracy': accuracy,
                'critical_weight': w_crit,
                **config
            })
    
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lsm_single_param_{param_name}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n{'='*70}")
        print(f"Best value for {param_name}:")
        best = df.loc[df['accuracy'].idxmax()]
        print(f"  Value: {best['value']}")
        print(f"  Accuracy: {best['accuracy']*100:.2f}%")
        print(f"\nResults saved to: {filename}")
        
        # Generate plot
        plot_single_param_results(df, param_name, timestamp)
        
        return df
    else:
        print("\n⚠️  No valid configurations found!")
        return None

def run_all_single_params(use_both_features=True):
    """Test all parameters one at a time and create comprehensive visualizations."""
    print("="*70)
    print("COMPREHENSIVE SINGLE PARAMETER ANALYSIS")
    print("="*70)
    
    feature_info = "mean_spike_times + spike_counts" if use_both_features else "mean_spike_times only"
    print(f"\nUsing features: {feature_info}")
    print(f"Testing {len(PARAM_RANGES)} parameters\n")
    
    all_results = []
    
    for param_name in PARAM_RANGES.keys():
        print(f"\n{'='*70}")
        print(f"Testing parameter: {param_name}")
        print(f"{'='*70}")
        
        df = run_single_param_sweep(param_name, use_both_features)
        if df is not None:
            all_results.append(df)
    
    if all_results:
        # Combine all results
        all_results_df = pd.concat(all_results, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save combined results
        combined_filename = f"lsm_all_params_results_{timestamp}.csv"
        all_results_df.to_csv(combined_filename, index=False)
        
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        # Create visualizations
        plot_all_params_heatmap(all_results_df, timestamp)
        plot_param_comparison(all_results_df, timestamp)
        
        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY OF BEST VALUES")
        print(f"{'='*70}")
        
        for param in PARAM_RANGES.keys():
            param_data = all_results_df[all_results_df['parameter'] == param]
            if not param_data.empty:
                best = param_data.loc[param_data['accuracy'].idxmax()]
                print(f"\n{param}:")
                print(f"  Best value: {best['value']}")
                print(f"  Accuracy: {best['accuracy']*100:.2f}%")
        
        print(f"\n{'='*70}")
        print(f"✅ All results saved to: {combined_filename}")
        print(f"✅ Visualizations generated")
        print(f"{'='*70}")

def main():
    """Main function - choose your search strategy."""
    
    # OPTION 1: Test ALL parameters (one at a time) with visualizations
    run_all_single_params(use_both_features=USE_BOTH_FEATURES)
    
    # OPTION 2: Test just one specific parameter
    # run_single_param_sweep('membrane_threshold', use_both_features=USE_BOTH_FEATURES)
    
    # OPTION 3: Test multiple specific parameters
    # for param in ['num_neurons', 'refractory_period', 'membrane_threshold']:
    #     run_single_param_sweep(param, use_both_features=USE_BOTH_FEATURES)

if __name__ == "__main__":
    main()