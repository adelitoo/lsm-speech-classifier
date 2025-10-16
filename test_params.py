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

# Parameter ranges to sample from
PARAM_RANGES = {
    'num_neurons': [1000, 1500, 2000, 2500, 3000, 3500, 4000],
    'num_output_neurons': [200, 300, 400, 500, 600, 700, 800],
    'refractory_period': [3, 4, 5, 6, 7, 8, 10],
    'membrane_threshold': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    'small_world_p': [0.1, 0.15,0.2, 0.25, 0.3, 0.35, 0.4],
    'small_world_k': [100, 150, 200, 250, 300, 350, 400],
}

# Base configuration
BASE_CONFIG = {
    'num_neurons': 3000,
    'num_output_neurons': 600,
    'leak_coefficient': 0,
    'refractory_period': 7,
    'membrane_threshold': 2.0,
    'small_world_p': 0.3,
    'small_world_k': 200
}

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    """Load the spike train dataset."""
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    return data['X_spikes'], data['y_labels']

def extract_features(lsm, spike_data, desc=""):
    """Extract mean spike time features from LSM."""
    features = []
    for sample in tqdm(spike_data, desc=desc, leave=False):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        feature_dict = lsm.extract_features_from_spikes()
        current_features = feature_dict["mean_spike_times"]
        current_features[current_features < 0] = 0
        features.append(current_features)
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

def test_configuration(config, X_train, X_test, y_train, y_test):
    """Test a single parameter configuration and return accuracy."""
    try:
        # Calculate critical weight for this configuration
        X_full = np.concatenate([X_train, X_test], axis=0)
        weights_mean = calculate_critical_weight(X_full, config)
        
        # Create SNN with current configuration
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
        
        # Extract features
        X_train_features = extract_features(lsm, X_train, desc="Train")
        X_test_features = extract_features(lsm, X_test, desc="Test")
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_train_features, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, weights_mean
    
    except Exception as e:
        print(f"Error with config: {e}")
        return None, None

def generate_random_config(params_to_vary=None):
    """
    Generate a random configuration by sampling from parameter ranges.
    
    Args:
        params_to_vary: List of parameter names to vary. If None, varies all parameters.
    """
    config = BASE_CONFIG.copy()
    
    if params_to_vary is None:
        params_to_vary = list(PARAM_RANGES.keys())
    
    for param in params_to_vary:
        if param in PARAM_RANGES:
            config[param] = random.choice(PARAM_RANGES[param])
    
    return config

def run_random_search(n_iterations=20, params_to_vary=None, seed=42):
    """
    Run random parameter search.
    
    Args:
        n_iterations: Number of random configurations to test
        params_to_vary: List of parameters to vary (None = all)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("="*70)
    print("LSM RANDOM PARAMETER SEARCH")
    print("="*70)
    
    if params_to_vary:
        print(f"\nVarying parameters: {', '.join(params_to_vary)}")
    else:
        print("\nVarying all parameters")
    
    print(f"Number of configurations to test: {n_iterations}")
    
    # Load data
    print("\nLoading dataset...")
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples\n")
    
    results = []
    best_accuracy = 0
    best_config = None
    
    # Test random configurations
    for i in range(n_iterations):
        config = generate_random_config(params_to_vary)
        
        print(f"\n{'='*70}")
        print(f"Configuration {i+1}/{n_iterations}")
        print(f"{'='*70}")
        print(f"Parameters:")
        for param, value in config.items():
            print(f"  {param}: {value}")
        
        accuracy, w_crit = test_configuration(config, X_train, X_test, y_train, y_test)
        
        if accuracy is not None:
            print(f"\n  → Accuracy: {accuracy*100:.2f}% | Critical weight: {w_crit:.6f}")
            
            result = {
                'iteration': i+1,
                'accuracy': accuracy,
                'critical_weight': w_crit,
                **config
            }
            results.append(result)
            
            # Track best configuration
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config.copy()
                print(f"  ★ NEW BEST ACCURACY! ★")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lsm_random_search_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"\nResults saved to: {filename}")
        
        # Print top 5 configurations
        df_sorted = df.sort_values('accuracy', ascending=False)
        print("\nTop 5 configurations:")
        print("-" * 70)
        for idx, row in df_sorted.head(5).iterrows():
            print(f"\nRank {idx+1} - Accuracy: {row['accuracy']*100:.2f}%")
            print(f"  num_neurons: {row['num_neurons']}")
            print(f"  num_output_neurons: {row['num_output_neurons']}")
            print(f"  refractory_period: {row['refractory_period']}")
            print(f"  membrane_threshold: {row['membrane_threshold']}")
            print(f"  small_world_p: {row['small_world_p']}")
            print(f"  small_world_k: {row['small_world_k']}")
            print(f"  leak_coefficient: {row['leak_coefficient']}")
            print(f"  critical_weight: {row['critical_weight']:.6f}")
        
        # Print best configuration details
        print("\n" + "="*70)
        print("BEST CONFIGURATION")
        print("="*70)
        if best_config:
            print(f"\nBest Accuracy: {best_accuracy*100:.2f}%")
            print("\nOptimal Parameters:")
            for param, value in best_config.items():
                print(f"  {param}: {value}")
        
        # Statistical summary
        print("\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)
        print(f"\nMean Accuracy: {df['accuracy'].mean()*100:.2f}%")
        print(f"Std Accuracy: {df['accuracy'].std()*100:.2f}%")
        print(f"Min Accuracy: {df['accuracy'].min()*100:.2f}%")
        print(f"Max Accuracy: {df['accuracy'].max()*100:.2f}%")

def run_single_param_sweep(param_name):
    """Test one parameter at a time (original functionality)."""
    print("="*70)
    print(f"TESTING PARAMETER: {param_name}")
    print("="*70)
    
    # Load data
    print("\nLoading dataset...")
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples\n")
    
    results = []
    
    if param_name not in PARAM_RANGES:
        print(f"Error: {param_name} not in parameter ranges")
        return
    
    for value in PARAM_RANGES[param_name]:
        config = BASE_CONFIG.copy()
        config[param_name] = value
        
        print(f"\n{param_name} = {value}")
        accuracy, w_crit = test_configuration(config, X_train, X_test, y_train, y_test)
        
        if accuracy is not None:
            print(f"  → Accuracy: {accuracy*100:.2f}% | Critical weight: {w_crit:.6f}")
            results.append({
                'parameter': param_name,
                'value': value,
                'accuracy': accuracy,
                'critical_weight': w_crit,
                **config
            })
    
    # Save and display results
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

def main():
    """Main function - choose your search strategy."""
    
    # OPTION 1: Random search - test multiple parameters together (RECOMMENDED)
    # Tests 20 random combinations of ALL parameters
    run_random_search(n_iterations=20, params_to_vary=None, seed=42)
    
    # OPTION 2: Random search - test specific parameters only
    # run_random_search(n_iterations=15, params_to_vary=['num_neurons', 'membrane_threshold', 'refractory_period'])
    
    # OPTION 3: Single parameter sweep (one at a time)
    # run_single_param_sweep('membrane_threshold')
    
    # OPTION 4: Test multiple single parameters
    # for param in ['num_neurons', 'refractory_period', 'membrane_threshold']:
    #     run_single_param_sweep(param)

if __name__ == "__main__":
    main()