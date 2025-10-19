import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import optuna
from datetime import datetime
import json
import pandas as pd

# Parameter search spaces (min, max, step for integers; min, max for floats)
PARAM_SEARCH_SPACE = {
    'num_neurons': (1000, 5000, 200),  # min, max, step
    'num_output_neurons': (300, 1000, 100),
    'refractory_period': (1, 12, 1),
    'membrane_threshold': (1.0, 5.0),  # continuous
    'small_world_p': (0.1, 0.6),  # continuous
    'small_world_k': (100, 500, 50),  # will be made even
}

# Feature combinations to try
# Available features from LSM: spike_counts, spike_variances, mean_spike_times, 
# first_spike_times, last_spike_times, mean_isi, isi_variances, burst_counts
FEATURE_COMBINATIONS = [
    ['mean_spike_times'],
    ['spike_counts'],
    ['spike_counts', 'mean_spike_times'],
    ['spike_counts', 'spike_variances'],
    ['mean_spike_times', 'first_spike_times'],
    ['spike_counts', 'mean_spike_times', 'spike_variances'],
    ['spike_counts', 'mean_spike_times', 'mean_isi'],
    ['spike_counts', 'mean_spike_times', 'first_spike_times', 'last_spike_times'],
    ['spike_counts', 'mean_spike_times', 'spike_variances', 'mean_isi'],
    ['spike_counts', 'mean_spike_times', 'burst_counts'],
]

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    'leak_coefficient': 0
}

# Global variables to store data (to avoid reloading)
X_TRAIN = None
X_TEST = None
Y_TRAIN = None
Y_TEST = None

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    """Load the spike train dataset."""
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    return data['X_spikes'], data['y_labels']

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
        
        # Extract and combine the specified features
        sample_feature_parts = []
        for feature_name in features_to_extract:
            # Get the feature vector from the dictionary
            feature_vector = feature_dict[feature_name].copy()
            
            # Clean up potential invalid values (NaNs, inf, negatives)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_vector[feature_vector < 0] = 0
            
            sample_feature_parts.append(feature_vector)
        
        # Concatenate all parts into one vector
        combined_features = np.concatenate(sample_feature_parts)
        all_combined_features.append(combined_features)

    return np.array(all_combined_features)

def calculate_critical_weight(X_spikes, config):
    """Calculate critical weight for the given configuration."""
    num_samples = X_spikes.shape[0]
    num_input_neurons = X_spikes.shape[1]
    num_time_bins = X_spikes.shape[2]
    
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    beta = config['small_world_k'] / 2
    w_critico = (config['membrane_threshold'] - 2 * I * config['refractory_period']) / beta
    return w_critico

def objective(trial):
    """
    Objective function for Optuna optimization.
    Returns accuracy to MAXIMIZE.
    """
    # Sample parameters from the search space
    config = {
        'num_neurons': trial.suggest_int('num_neurons', PARAM_SEARCH_SPACE['num_neurons'][0], 
                                         PARAM_SEARCH_SPACE['num_neurons'][1], 
                                         step=PARAM_SEARCH_SPACE['num_neurons'][2]),
        'num_output_neurons': trial.suggest_int('num_output_neurons', PARAM_SEARCH_SPACE['num_output_neurons'][0],
                                                PARAM_SEARCH_SPACE['num_output_neurons'][1],
                                                step=PARAM_SEARCH_SPACE['num_output_neurons'][2]),
        'refractory_period': trial.suggest_int('refractory_period', PARAM_SEARCH_SPACE['refractory_period'][0],
                                               PARAM_SEARCH_SPACE['refractory_period'][1],
                                               step=PARAM_SEARCH_SPACE['refractory_period'][2]),
        'membrane_threshold': trial.suggest_float('membrane_threshold', *PARAM_SEARCH_SPACE['membrane_threshold']),
        'small_world_p': trial.suggest_float('small_world_p', *PARAM_SEARCH_SPACE['small_world_p']),
    }
    
    # Add fixed parameters
    config.update(FIXED_PARAMS)
    
    # Sample k and ensure it's even
    k_raw = trial.suggest_int('small_world_k_raw', PARAM_SEARCH_SPACE['small_world_k'][0],
                              PARAM_SEARCH_SPACE['small_world_k'][1],
                              step=PARAM_SEARCH_SPACE['small_world_k'][2])
    config['small_world_k'] = k_raw if k_raw % 2 == 0 else k_raw + 1
    
    # Ensure num_output_neurons < num_neurons to avoid error
    if config['num_output_neurons'] >= config['num_neurons']:
        config['num_output_neurons'] = config['num_neurons'] - 100
        if config['num_output_neurons'] < 100:
            # Configuration is invalid
            return 0.0
    
    # Sample which features to use
    feature_idx = trial.suggest_categorical('feature_combination_idx', list(range(len(FEATURE_COMBINATIONS))))
    features_to_extract = FEATURE_COMBINATIONS[feature_idx]
    
    try:
        # Calculate critical weight for this configuration
        X_full = np.concatenate([X_TRAIN, X_TEST], axis=0)
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
            input_spike_times=X_TRAIN[0]
        )
        
        lsm = SNN(simulation_params=params)
        
        # Extract features with selected combination
        X_train_features = extract_features(lsm, X_TRAIN, features_to_extract, desc=f"Trial {trial.number}")
        X_test_features = extract_features(lsm, X_TEST, features_to_extract, desc=f"Trial {trial.number}")
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_train_features, Y_TRAIN)
        
        # Evaluate
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(Y_TEST, y_pred)
        
        # Store additional info
        trial.set_user_attr('critical_weight', weights_mean)
        trial.set_user_attr('features_used', str(features_to_extract))
        
        print(f"\nTrial {trial.number}: Accuracy = {accuracy*100:.2f}% | Features: {features_to_extract}")
        
        return accuracy
    
    except Exception as e:
        print(f"\nTrial {trial.number} failed: {e}")
        return 0.0

def run_bayesian_optimization(n_trials=50, timeout=None):
    """
    Run Bayesian optimization using Optuna.
    
    Args:
        n_trials: Number of trials to run
        timeout: Maximum time in seconds (None = no limit)
    """
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    
    print("="*70)
    print("LSM BAYESIAN OPTIMIZATION (Optuna)")
    print("="*70)
    
    # Load data once
    print("\nLoading dataset...")
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Dataset loaded: {len(X_TRAIN)} train, {len(X_TEST)} test samples")
    
    print(f"\nFeature combinations to test:")
    for i, combo in enumerate(FEATURE_COMBINATIONS):
        print(f"  {i}: {combo}")
    
    print(f"\nStarting Bayesian optimization with {n_trials} trials...")
    print("Optuna will intelligently search for optimal parameters AND features.\n")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f'lsm_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Accuracy: {study.best_value*100:.2f}%")
    
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        if param == 'small_world_k_raw':
            actual_k = value if value % 2 == 0 else value + 1
            print(f"  small_world_k: {actual_k}")
        elif param == 'feature_combination_idx':
            features = FEATURE_COMBINATIONS[value]
            print(f"  features: {features}")
        else:
            print(f"  {param}: {value}")
    
    print(f"\nCritical Weight: {study.best_trial.user_attrs['critical_weight']:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study
    study_filename = f"lsm_bayesian_study_{timestamp}.pkl"
    optuna.save_study(study, study_filename)
    print(f"\nStudy saved to: {study_filename}")
    
    # Save best parameters to JSON
    best_params = study.best_params.copy()
    if 'small_world_k_raw' in best_params:
        k_raw = best_params.pop('small_world_k_raw')
        best_params['small_world_k'] = k_raw if k_raw % 2 == 0 else k_raw + 1
    
    if 'feature_combination_idx' in best_params:
        feature_idx = best_params.pop('feature_combination_idx')
        best_params['features'] = FEATURE_COMBINATIONS[feature_idx]
    
    best_config = {
        'best_accuracy': study.best_value,
        'best_trial': study.best_trial.number,
        'critical_weight': study.best_trial.user_attrs['critical_weight'],
        'parameters': best_params
    }
    
    json_filename = f"lsm_best_config_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Best configuration saved to: {json_filename}")
    
    # Save all trials to CSV
    trials_data = []
    for trial in study.trials:
        if trial.value is None:
            continue
        
        trial_dict = {
            'trial_number': trial.number,
            'accuracy': trial.value,
            'critical_weight': trial.user_attrs.get('critical_weight', None),
            'features_used': trial.user_attrs.get('features_used', None)
        }
        
        # Add parameters
        for param, value in trial.params.items():
            if param == 'small_world_k_raw':
                trial_dict['small_world_k'] = value if value % 2 == 0 else value + 1
            elif param == 'feature_combination_idx':
                trial_dict['feature_combination'] = str(FEATURE_COMBINATIONS[value])
            else:
                trial_dict[param] = value
        
        # Add fixed parameters
        trial_dict['leak_coefficient'] = FIXED_PARAMS['leak_coefficient']
        
        trials_data.append(trial_dict)
    
    df = pd.DataFrame(trials_data)
    csv_filename = f"lsm_bayesian_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"All trial results saved to: {csv_filename}")
    
    # Top 5 trials
    print("\n" + "="*70)
    print("TOP 5 TRIALS")
    print("="*70)
    
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for i, trial in enumerate(top_trials, 1):
        if trial.value is None:
            continue
        print(f"\n#{i} - Trial {trial.number}: Accuracy = {trial.value*100:.2f}%")
        print(f"  Features: {trial.user_attrs.get('features_used', 'N/A')}")
        print(f"  Parameters:")
        for param, value in trial.params.items():
            if param == 'small_world_k_raw':
                actual_k = value if value % 2 == 0 else value + 1
                print(f"    small_world_k: {actual_k}")
            elif param == 'feature_combination_idx':
                print(f"    feature_combo: {FEATURE_COMBINATIONS[value]}")
            else:
                print(f"    {param}: {value}")
    
    # Feature combination analysis
    print("\n" + "="*70)
    print("FEATURE COMBINATION PERFORMANCE")
    print("="*70)
    
    feature_performance = {}
    for trial in study.trials:
        if trial.value is None:
            continue
        if 'feature_combination_idx' in trial.params:
            feat_combo = str(FEATURE_COMBINATIONS[trial.params['feature_combination_idx']])
            if feat_combo not in feature_performance:
                feature_performance[feat_combo] = []
            feature_performance[feat_combo].append(trial.value)
    
    print("\nAverage accuracy by feature combination:")
    for feat_combo, accuracies in sorted(feature_performance.items(), 
                                         key=lambda x: np.mean(x[1]), 
                                         reverse=True):
        avg_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        print(f"  {feat_combo}")
        print(f"    Avg: {avg_acc*100:.2f}% | Max: {max_acc*100:.2f}% | Trials: {len(accuracies)}")
    
    # Parameter importance
    print("\n" + "="*70)
    print("PARAMETER IMPORTANCE")
    print("="*70)
    print("\n(How much each parameter affects accuracy)")
    
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            if param == 'small_world_k_raw':
                param = 'small_world_k'
            elif param == 'feature_combination_idx':
                param = 'feature_combination'
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"  Could not calculate importance: {e}")
    
    # Optimization history
    print("\n" + "="*70)
    print("OPTIMIZATION PROGRESS")
    print("="*70)
    
    accuracies = [t.value*100 for t in study.trials if t.value is not None]
    if accuracies:
        print(f"\nFirst trial accuracy: {accuracies[0]:.2f}%")
        print(f"Best accuracy found: {max(accuracies):.2f}%")
        print(f"Improvement: +{max(accuracies) - accuracies[0]:.2f}%")
    
    return study

def continue_optimization(study_filename, n_additional_trials=20):
    """
    Continue optimization from a saved study.
    
    Args:
        study_filename: Path to saved study (.pkl file)
        n_additional_trials: Number of additional trials to run
    """
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    
    print(f"Loading study from {study_filename}...")
    study = optuna.load_study(study_filename)
    
    print(f"Continuing optimization with {n_additional_trials} additional trials...")
    print(f"Current best accuracy: {study.best_value*100:.2f}%")
    
    # Load data
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Continue optimization
    study.optimize(objective, n_trials=n_additional_trials, show_progress_bar=True)
    
    print(f"\nNew best accuracy: {study.best_value*100:.2f}%")
    
    return study

def main():
    """Main function."""
    
    study = run_bayesian_optimization(n_trials=100)

if __name__ == "__main__":
    main()