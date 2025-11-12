import os
import argparse


def run_pipeline(
        n_filters: int,
        filterbank: str,
        feature_set: str,
        multiplier: float):
    """
    Runs the entire pipeline:
    1. Creates the spike train dataset.
    2. Extracts features with the LSM.
    3. Trains and evaluates the final classifier.
    """
    print("--- Running Pipeline ---")

    print("\n--- Step 1: Creating Spike Train Dataset ---")
    os.system(
        f"python create_dataset.py --n-filters {n_filters} --filterbank {filterbank}")

    print("\n--- Step 2: Extracting LSM Features ---")
    os.system(
        f"python extract_lsm_features.py --feature-set {feature_set} --multiplier {multiplier}")

    print("\n--- Step 3: Training and Evaluating Classifier ---")
    os.system("python train_classifier.py")

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the entire speech recognition pipeline.")
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
    parser.add_argument(
        "--feature-set",
        type=str,
        default="original",
        choices=[
            'all',
            'rate',
            'timing',
            'rhythm',
            'original'],
        help="The set of features to extract.")
    parser.add_argument("--multiplier", type=float, default=0.6,
                        help="Multiplier for w_critico.")

    args = parser.parse_args()

    run_pipeline(
        n_filters=args.n_filters,
        filterbank=args.filterbank,
        feature_set=args.feature_set,
        multiplier=args.multiplier
    )
