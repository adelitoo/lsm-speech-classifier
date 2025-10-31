#!/bin/bash


set -e

echo "Dataset creation..."
python spect_to_spike_train.py

echo "Extracting features..."
python extract_lsm_features.py

echo "Training readout layer..."
python train_classifier.py
