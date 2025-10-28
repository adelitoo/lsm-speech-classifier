# Project Overview

This project is a speech recognition pipeline that uses a Spiking Neural Network (SNN), specifically a Liquid State Machine (LSM), to classify spoken words from the Google Speech Commands v0.02 dataset. The pipeline transforms raw audio signals into high-dimensional feature representations that are then classified by a standard machine learning model.

The project is written in Python and uses the following main libraries:
- `snnpy`: For the Liquid State Machine implementation.
- `librosa`: For audio processing and feature extraction (Mel spectrograms).
- `scikit-learn`: For the final classification step (Logistic Regression).
- `numpy`: For numerical operations.

The pipeline is divided into three main stages:
1.  **Dataset Creation**: Converts raw audio files into spike trains.
2.  **Feature Extraction**: Processes the spike trains through the LSM to generate feature vectors.
3.  **Classification**: Trains a classifier on the extracted features and evaluates its performance.

# Building and Running

## 1. Installation

Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

## 2. Dataset

1.  Download the **Google Speech Commands v0.02** dataset.
2.  Extract the dataset and ensure the directory containing the spoken words (e.g., `yes`, `no`, `up`, `down`) is named `speech_commands_v0.02` and placed in the project's root folder.

## 3. Running the Pipeline

Execute the scripts from your terminal in the following order.

### Step 1: Create the Spike Train Dataset

This script converts all the audio files. It only needs to be run once, or whenever you change the spike encoding parameters.

```bash
python spect_to_spike_train.py
```

### Step 2: Extract Features with the LSM

This script processes the spike trains through the LSM. Run this script after creating the dataset, or whenever you change the LSM parameters.

```bash
python extract_lsm_features.py
```

### Step 3: Train and Evaluate the Final Classifier

This script trains the classifier on the extracted features and gives you the final result.

```bash
python train_classifier.py
```

# Development Conventions

- The code is structured into a clear three-stage pipeline.
- Each script has a clear purpose and is well-documented.
- The project uses a `requirements.txt` file to manage dependencies.
- The code uses type hints.
- The project includes a `README.md` file with clear instructions.
