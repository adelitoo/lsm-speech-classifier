import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
import argparse


NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 2
MEMBRANE_THRESHOLD = 2.0
SMALL_WORLD_P = 0.1
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2)


FEATURE_SETS = {
    'all': ['spike_counts', 'spike_variances', 'mean_spike_times',
            'first_spike_times', 'last_spike_times', 'mean_isi',
            'isi_variances', 'burst_counts'],
    'rate': ['spike_counts', 'spike_variances', 'burst_counts'],
    'timing': ['mean_spike_times', 'first_spike_times', 'last_spike_times'],
    'rhythm': ['mean_isi', 'isi_variances'],
    'original': ['spike_counts', 'spike_variances', 'mean_spike_times',
                 'mean_isi', 'isi_variances']
}

np.random.seed(42)


def calculate_theoretical_w_critico(lsm_params, input_data):
    """
    Calculates the theoretical w_critico based on network params and input data.
    The formula is derived from the mean-field approximation of the network dynamics.
    It estimates the critical weight at which the network is at the edge of chaos,
    which is often where computation is maximized.
    """
    num_samples = min(500, len(input_data))
    total_spikes = np.sum([np.sum(sample)
                          for sample in input_data[:num_samples]])
    total_elements = np.sum([sample.shape[0] * sample.shape[1]
                            for sample in input_data[:num_samples]])

    if total_elements == 0:
        return 0.007

    avg_I = total_spikes / total_elements
    beta = lsm_params.small_world_graph_k / 2

    if beta == 0:
        return 0.007

    numerator = (lsm_params.membrane_threshold - 2 *
                 avg_I * (lsm_params.refractory_period))
    w_critico = numerator / beta

    print(f"Theoretical w_critico: {w_critico:.8f}")
    return w_critico


def load_spike_dataset(filename="speech_spike_dataset_pure_redundancy.npz"):
    if not Path(filename).exists():
        print(f"Error: Dataset not found at '{filename}'")
        return None, None

    data = np.load(filename)
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']

    print(f"Loaded {len(X_spikes)} samples from '{filename}'")
    return X_spikes, y_labels


def extract_all_features(lsm, spike_data, feature_keys, desc=""):
    all_features = []
    for sample in tqdm(spike_data, desc=desc):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()

        feature_dict = lsm.extract_features_from_spikes()

        parts = [np.nan_to_num(feature_dict[key].copy())
                 for key in feature_keys if key in feature_dict]
        all_features.append(np.concatenate(parts))

    return np.array(all_features)


def main(feature_set: str, multiplier: float):
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    base_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0]
    )

    w_critico_calculated = calculate_theoretical_w_critico(
        base_params, X_train)
    optimal_weight = w_critico_calculated * multiplier

    print(f"Using weight: {optimal_weight:.8f} (multiplier: {multiplier:.2f})")

    base_params.mean_weight = optimal_weight
    base_params.weight_variance = optimal_weight * 0.1

    lsm = SNN(simulation_params=base_params)

    feature_keys = FEATURE_SETS[feature_set]
    print(f"Extracting feature set: '{feature_set}'")

    X_train_feat = extract_all_features(lsm, X_train, feature_keys, "Training")
    X_test_feat = extract_all_features(lsm, X_test, feature_keys, "Testing")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    output_file = "lsm_features_larger.npz"
    np.savez_compressed(
        output_file,
        X_train_features=X_train_scaled,
        y_train=y_train,
        X_test_features=X_test_scaled,
        y_test=y_test,
        feature_set=feature_set
    )

    print(f"Extraction complete. Features saved to '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from a spike train dataset using an LSM.")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="original",
        choices=FEATURE_SETS.keys(),
        help="The set of features to extract.")
    parser.add_argument("--multiplier", type=float, default=0.6,
                        help="Multiplier for w_critico.")

    args = parser.parse_args()
    main(feature_set=args.feature_set, multiplier=args.multiplier)
