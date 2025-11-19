# Spiking Neural Network for Speech Command Recognition

This project demonstrates a complete machine learning pipeline for classifying spoken words using a brain-inspired Spiking Neural Network (SNN), specifically a **Liquid State Machine (LSM)**. The goal is to transform raw audio signals into high-dimensional feature representations that can be easily classified by a standard machine learning model.

The model is trained on a subset of the Google Speech Commands v0.02 dataset to recognize four words: **"yes"**, **"no"**, **"up"**, and **"down"**.

---

## How to Run

Execute the main script from your terminal. You can customize the pipeline with command-line arguments.

```bash
python main.py --n-filters 128 --filterbank gammatone --feature-set original --multiplier 0.6
```

---

## Setup and Installation

### 1. Prerequisites

- Python 3.8+

### 2. Dataset

1. Download the **Google Speech Commands v0.02** dataset. [ http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz ]
2. Extract the dataset and ensure the directory containing the spoken words (e.g., `yes`, `no`, `up`, `down`) is named `speech_commands_v0.02` and placed in the project's root folder.

### 3. Dependencies

Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

---

## Results

With the latest configuration, the model achieves the following performance on the test set:

- **Final Test Accuracy:** `67.83%`

### **Classification Report:**

```
              precision    recall  f1-score   support

         yes       0.71      0.74      0.73       200
          no       0.60      0.66      0.63       200
          up       0.63      0.69      0.65       200
      visual       0.69      0.69      0.69       200
    backward       0.72      0.73      0.72       200
        stop       0.74      0.68      0.70       200
        bird       0.67      0.61      0.64       200
         cat       0.65      0.60      0.62       200
        nine       0.64      0.68      0.66       200
       eight       0.72      0.71      0.71       200
        zero       0.66      0.66      0.66       200
      follow       0.74      0.71      0.73       200

    accuracy                           0.68      2400
   macro avg       0.68      0.68      0.68      2400
weighted avg       0.68      0.68      0.68      2400
```

---

## License

This project is released under the MIT License.
Feel free to modify and use it for educational or research purposes.
