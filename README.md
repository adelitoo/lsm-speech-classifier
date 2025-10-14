# Spiking Neural Network for Speech Command Recognition

This project demonstrates a complete machine learning pipeline for classifying spoken words using a brain-inspired Spiking Neural Network (SNN), specifically a **Liquid State Machine (LSM)**. The goal is to transform raw audio signals into high-dimensional feature representations that can be easily classified by a standard machine learning model.

The model is trained on a subset of the Google Speech Commands v0.02 dataset to recognize four words: **"yes"**, **"no"**, **"up"**, and **"down"**.

---

## Features

- **Audio Processing:** Uses `librosa` to process audio files and convert them into Mel spectrograms.  
- **Spike Encoding:** Implements a threshold-based method to convert normalized spectrograms into sparse, temporal spike trains.  
- **Reservoir Computing:** Employs a Liquid State Machine (LSM) as a fixed, non-linear feature extractor. The LSM is built using the `snnpy` library.  
- **Criticality Tuning:** Automatically calculates and sets the LSM's average synaptic weight to a near-critical value, which is theorized to maximize its computational power.  
- **Classification:** Uses a `RandomForestClassifier` from `scikit-learn` to classify the features extracted by the LSM.

---

## Workflow

The project is divided into a three-stage pipeline, with each stage handled by a dedicated script:

### **Stage 1: Dataset Creation** (`create_dataset.py`)

- Takes the raw `.wav` audio files as input.  
- Converts each audio file into a normalized Mel spectrogram.  
- Encodes each spectrogram into a binary spike train.  
- Saves the complete dataset of spike trains and labels into a single compressed file (`speech_spike_dataset.npz`).

### **Stage 2: Feature Extraction** (`extract_lsm_features.py`)

- Loads the spike train dataset.  
- Initializes an LSM, tuning its synaptic weight to a calculated critical value.  
- Processes each spike train through the LSM to generate a feature vector (e.g., based on spike counts or mean spike times).  
- Saves the extracted feature vectors and labels into a new file (`lsm_features_dataset.npz`).

### **Stage 3: Classification** (`train_classifier.py`)

- Loads the final feature dataset.  
- Trains a Random Forest classifier on the training set.  
- Evaluates the classifier's performance on the test set.  
- Prints the final accuracy and a detailed classification report.

---

## Setup and Installation

### 1. Prerequisites

- Python 3.8+

### 2. Dataset

1. Download the **Google Speech Commands v0.02** dataset.  
2. Extract the dataset and ensure the directory containing the spoken words (e.g., `yes`, `no`, `up`, `down`) is named `speech_commands_v0.02` and placed in the project's root folder.

### 3. Dependencies

Install the required Python libraries using pip:

```bash
pip install numpy librosa scipy scikit-learn matplotlib tqdm snnpy
```

---

## How to Run

Execute the scripts from your terminal in the following order.

### **Step 1: Create the Spike Train Dataset**

This script converts all the audio files. It only needs to be run once, or whenever you change the spike encoding parameters.

```bash
python create_dataset.py
```

### **Step 2: Extract Features with the LSM**

This script processes the spike trains through the LSM. Run this script after creating the dataset, or whenever you change the LSM parameters.

```bash
python extract_lsm_features.py
```

### **Step 3: Train and Evaluate the Final Classifier**

This script trains the classifier on the extracted features and gives you the final result.

```bash
python train_classifier.py
```

---

## Results

With the latest configuration, the model achieves the following performance on the test set:

- **Final Test Accuracy:** `63.50%`

### **Classification Report:**

```
              precision    recall  f1-score   support

         yes       0.68      0.76      0.72       100
          no       0.51      0.46      0.48       100
          up       0.70      0.70      0.70       100
        down       0.63      0.62      0.63       100

    accuracy                           0.64       400
   macro avg       0.63      0.64      0.63       400
weighted avg       0.63      0.64      0.63       400
```

---

## License

This project is released under the MIT License.  
Feel free to modify and use it for educational or research purposes.
